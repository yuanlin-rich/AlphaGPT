# AlphaGPT 项目分析文档

## 项目概述

**AlphaGPT** 是一个面向 Solana meme 代币生态的自动化量化交易系统。其核心创新在于使用 Transformer 模型自动生成可解释的因子公式，通过回测评分筛选出高性能策略，并集成到实盘交易执行链路中。

### 核心设计理念
- **不是直接预测价格**，而是"生成公式 → 解释执行 → 回测评分 → 优化生成器"的自动化策略发现流程
- **公式 = token 序列**：由特征（因子）和算子组成，通过 StackVM 执行生成因子信号
- **清晰的分层架构**：将策略研究与交易执行分离，提高系统可维护性和扩展性

## 系统架构总览

```
AlphaGPT 系统架构
├── data_pipeline/      # 数据管道 - 数据获取与存储
├── model_core/         # 模型核心 - 策略生成与回测
├── strategy_manager/   # 策略管理 - 实盘执行与风控
├── execution/          # 交易执行 - 链上交易
├── dashboard/          # 仪表板 - 监控与控制
└── 实验与研究文件
```

## 详细模块分析

### 1. 数据管道 (`data_pipeline/`)

#### 功能
从 Birdeye/DexScreener API 获取 Solana meme 代币的行情数据（OHLCV），经过过滤后存储到 PostgreSQL/TimescaleDB 数据库。

#### 关键组件
- **`config.py`** - 数据管道配置
  ```python
  DB_USER = os.getenv("DB_USER", "postgres")
  DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
  DB_HOST = os.getenv("DB_HOST", "localhost")
  DB_PORT = os.getenv("DB_PORT", "5432")
  DB_NAME = os.getenv("DB_NAME", "crypto_quant")
  MIN_LIQUIDITY_USD = 500000.0  # 最小流动性
  MIN_FDV = 10000000.0          # 最小完全稀释估值
  BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
  ```

- **`fetcher.py`** - 异步数据获取器
  - 使用 aiohttp 并发获取数据
  - 支持 Birdeye 和 DexScreener 双数据源
  - 实现请求限流和错误处理

- **`data_manager.py`** - 数据管理主类
  - 执行完整的数据同步流程
  - 过滤低流动性/高市值代币
  - 批量插入数据库

- **`db_manager.py`** - 数据库操作
  - 使用 SQLAlchemy + asyncpg 异步操作 PostgreSQL
  - 管理 tokens 和 ohlcv 表

- **`run_pipeline.py`** - 管道入口脚本
  - 启动每日数据同步任务

#### 数据流程
1. 获取 trending tokens (限制：付费 API 500个，免费 100个)
2. 过滤：流动性 > 50万美元，FDV 在 1000万到无穷大
3. 存储代币元数据到数据库
4. 获取每个代币的 OHLCV 历史数据（默认7天）
5. 存储 OHLCV 数据

### 2. 模型核心 (`model_core/`)

#### 功能
使用 Transformer 模型生成交易策略公式，通过回测评估公式性能，训练模型生成更优公式。

#### 关键组件

##### **`alphagpt.py`** - 核心模型
- **NewtonSchulzLowRankDecay**：低秩衰减正则化（LoRD）
  - 使用 Newton-Schulz 迭代计算最小奇异向量
  - 针对 attention 和 key 参数的低秩结构正则化
- **StableRankMonitor**：稳定秩监控器
  - 监控模型参数的有效秩
- **AlphaGPT 模型架构**：
  - RMSNorm：均方根层归一化
  - QKNorm：查询-键归一化
  - SwiGLU：Swish GLU 激活函数
  - MTPHead：多任务池化头

##### **`engine.py`** - 训练引擎
```python
class AlphaEngine:
    def __init__(self, use_lord_regularization=True):
        self.loader = CryptoDataLoader()    # 数据加载
        self.model = AlphaGPT()             # 模型
        self.opt = torch.optim.AdamW()      # 优化器
        self.lord_opt = NewtonSchulzLowRankDecay()  # LoRD 正则化
        self.vm = StackVM()                 # 公式执行虚拟机
        self.bt = MemeBacktest()            # 回测器
```

##### **`factors.py`** - 特征工程
- **基础因子（6个）**：
  - `ret`：对数收益
  - `liq_score`：流动性/FDV 健康度
  - `pressure`：买卖力量不平衡
  - `fomo`：成交量加速度
  - `dev`：价格偏离均值
  - `log_vol`：对数成交量

- **扩展因子（12个）**：
  - 包含基础因子 + `vol_cluster`、`momentum_rev`、`rel_strength`、`hl_range` 等

##### **`ops.py`** - 算子定义
```python
OPS_CONFIG = {
    "ADD": lambda x, y: x + y,      # 加法
    "SUB": lambda x, y: x - y,      # 减法
    "MUL": lambda x, y: x * y,      # 乘法
    "DIV": lambda x, y: x / (y + 1e-9),  # 除法
    "NEG": lambda x: -x,            # 取负
    "ABS": lambda x: abs(x),        # 绝对值
    "SIGN": lambda x: 1 if x > 0 else -1 if x < 0 else 0,  # 符号
    "GATE": lambda c, x, y: x if c > 0 else y,  # 门控选择
    "JUMP": lambda x: 1 if abs(x) > 3 else 0,   # 极端跳变检测
    "DECAY": lambda t, lag1, lag2: t + 0.8*lag1 + 0.6*lag2,  # 衰减叠加
    "DELAY1": lambda x: x,          # 滞后1（占位符）
    "MAX3": lambda x, y, z: max(x, y, z)  # 三值最大值
}
```

##### **`vm.py`** - StackVM
- 执行公式 token 序列的栈式虚拟机
- 将因子和算子组合计算最终信号分数

##### **`backtest.py`** - 回测引擎
- 针对 meme 代币特性的回测器
- 考虑流动性、滑点、交易费用
- 输出夏普比率、最大回撤等指标

##### **`config.py`** - 模型配置
```python
class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0
    BASE_FEE = 0.005  # 0.5% (Swap + Gas + Jito Tip)
```

#### 训练流程
1. 加载历史数据
2. 生成随机公式 token 序列
3. 使用 StackVM 执行公式计算信号
4. 回测评估公式性能（夏普比率等）
5. 使用性能作为奖励训练 Transformer
6. 应用 LoRD 正则化防止过拟合
7. 保存最佳公式到 `best_meme_strategy.json`

### 3. 策略管理 (`strategy_manager/`)

#### 功能
加载训练好的策略公式，实时扫描市场，生成交易信号，管理持仓和风险。

#### 关键组件

##### **`runner.py`** - 策略运行器
- 主运行循环，周期性执行：
  1. 加载最新市场数据
  2. 对代币计算公式得分
  3. 应用风险过滤
  4. 生成买卖信号
  5. 调用交易执行

##### **`portfolio.py`** - 持仓管理
```python
@dataclass
class Position:
    token_address: str      # 代币地址
    symbol: str            # 代币符号
    entry_price: float     # 入场价格
    entry_time: float      # 入场时间戳
    amount_held: float     # 持仓数量
    initial_cost_sol: float # 初始投入 SOL
    highest_price: float   # 最高价（用于计算回撤）
    is_moonbag: bool = False  # 是否已出本，利润奔跑
```

- **持仓策略**：
  - 止盈止损：-5% 止损，+10% 部分止盈
  - 移动止盈：激活后跟踪回撤
  - Moonbag 策略：翻倍后卖出50%，剩余持仓让利润奔跑

##### **`risk.py`** - 风险引擎
- 流动性过滤
- 持仓集中度控制
- 最大回撤限制

##### **`config.py`** - 策略配置
```python
class StrategyConfig:
    MAX_OPEN_POSITIONS = 3          # 最大同时持仓数
    ENTRY_AMOUNT_SOL = 2.0          # 每笔交易投入 SOL
    STOP_LOSS_PCT = -0.05           # 止损比例 -5%
    TAKE_PROFIT_Target1 = 0.10      # 第一目标止盈 +10%
    TP_Target1_Ratio = 0.5          # 第一目标卖出比例 50%
    TRAILING_ACTIVATION = 0.05      # 移动止盈激活阈值 +5%
    TRAILING_DROP = 0.03            # 移动止盈回撤阈值 -3%
    BUY_THRESHOLD = 0.85            # 买入信号阈值
    SELL_THRESHOLD = 0.45           # 卖出信号阈值
```

### 4. 交易执行 (`execution/`)

#### 功能
在 Solana 区块链上执行实际的买卖交易，集成 Jupiter 聚合器获取最优价格。

#### 关键组件

##### **`trader.py`** - 交易执行器
```python
class SolanaTrader:
    async def buy(self, token_address: str, amount_sol: float, slippage_bps=500):
        # 1. 检查 SOL 余额
        # 2. 从 Jupiter 获取报价
        # 3. 构建并签名交易
        # 4. 发送并确认交易
    
    async def sell(self, token_address: str, percentage: float = 1.0, slippage_bps=500):
        # 1. 检查代币余额
        # 2. 从 Jupiter 获取报价
        # 3. 构建并签名交易
        # 4. 发送并确认交易
```

##### **`jupiter.py`** - Jupiter 聚合器集成
- 获取最优报价路由
- 构建 swap 交易
- 处理交易签名

##### **`rpc_handler.py`** - Solana RPC 客户端
- 与 Solana 节点通信
- 查询余额、发送交易、确认交易

##### **`config.py`** - 执行配置
```python
class ExecutionConfig:
    RPC_URL = os.getenv("QUICKNODE_RPC_URL", "填入RPC地址")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")  # 私钥
    PAYER_KEYPAIR = Keypair.from_base58_string(_PRIV_KEY_STR)
    WALLET_ADDRESS = str(PAYER_KEYPAIR.pubkey())
    DEFAULT_SLIPPAGE_BPS = 200  # 默认滑点 2%
    SOL_MINT = "So11111111111111111111111111111111111111112"  # SOL mint
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC mint
```

### 5. 仪表板 (`dashboard/`)

#### 功能
基于 Streamlit 的 Web 仪表板，用于实时监控系统状态、查看持仓、控制交易。

#### 关键组件

##### **`app.py`** - 主应用
- 实时显示关键指标：
  - 钱包余额
  - 当前持仓
  - 市场快照
  - 交易日志
- 控制功能：
  - 数据刷新
  - 紧急停止（创建 STOP_SIGNAL 文件）

##### **`data_service.py`** - 数据服务
- 从数据库和本地文件读取数据
- 提供仪表板所需的数据接口

##### **`visualizer.py`** - 可视化
- 使用 Plotly 绘制图表：
  - PnL 分布图
  - 市场散点图
  - 持仓变化图

### 6. 实验与研究文件

#### **`lord/experiment.py`**
- LoRD（低秩衰减）正则化实验
- 研究低秩正则化对模型性能的影响

#### **`times.py`**
- 中国股市回测实验（独立模块）
- 使用 tushare 获取A股数据
- 测试传统因子在股市的表现

#### **`paper/`** 目录
- 研究论文和文档
- `20251226.pdf`：可能的技术论文

## 技术栈详情

### 核心依赖（requirements.txt）
```txt
# 深度学习与数值计算
torch>=2.0.0          # PyTorch
numpy>=1.24.0         # 数值运算

# 数据处理与数据库
pandas>=2.0.0         # 数据分析
sqlalchemy>=2.0.0     # 数据库 ORM
asyncpg>=0.28.0       # 异步 PostgreSQL 驱动

# 异步 HTTP
aiohttp>=3.9.0        # 异步 HTTP 客户端

# 环境配置
python-dotenv>=1.0.0  # .env 文件加载

# 日志与进度
loguru>=0.7.0         # 结构化日志
tqdm>=4.66.0          # 进度条

# Solana 区块链集成
solders>=0.18.0       # Solana Rust 绑定
solana>=0.30.0        # Solana Python SDK
base58>=2.1.0         # Base58 编码

# 仪表板与可视化
streamlit>=1.28.0     # Web 仪表板框架
plotly>=5.17.0        # 交互式图表

# PostgreSQL
psycopg2-binary>=2.9.10
```

### 可选依赖（requirements-optional.txt）
```txt
matplotlib>=3.7.0     # 绘图（实验脚本）
seaborn>=0.12.0       # 统计可视化
tushare>=1.2.89       # 中国股市数据（times.py）
```

## 完整工作流程

### 阶段1：数据准备
```
1. 配置环境变量（数据库、API密钥、私钥）
2. 运行数据管道：python -m data_pipeline.run_pipeline
3. 数据入库：trending tokens → 过滤 → OHLCV 存储
```

### 阶段2：模型训练
```
1. 加载历史数据：CryptoDataLoader().load_data()
2. 初始化模型：AlphaGPT()
3. 训练循环：
   - 生成随机公式
   - StackVM 执行计算信号
   - 回测评估性能
   - 反向传播更新模型
   - 应用 LoRD 正则化
4. 保存最佳公式：best_meme_strategy.json
```

### 阶段3：实盘交易
```
1. 启动策略运行器：StrategyRunner().run_loop()
2. 周期性执行（例如每5分钟）：
   a. 扫描市场：获取最新代币数据
   b. 计算信号：对每个代币应用公式得分
   c. 风险过滤：流动性、市值等检查
   d. 生成交易决策：
      - 买入：得分 > 0.85 且有空仓位
      - 卖出：得分 < 0.45 或触发止盈止损
   e. 执行交易：通过 Jupiter 聚合器买卖
   f. 更新持仓：记录交易结果
3. 实时监控：通过仪表板查看状态
```

### 阶段4：监控与维护
```
1. 仪表板访问：streamlit run dashboard/app.py
2. 日志查看：loguru 结构化日志
3. 紧急干预：通过仪表板发送停止信号
4. 定期优化：重新训练模型更新策略
```

## 关键特性总结

### 创新点
1. **自动化策略发现**：使用 Transformer 自动生成交易公式，而非人工设计
2. **可解释性**：策略以数学公式形式呈现，易于理解和调整
3. **低秩正则化**：应用 LoRD 技术防止过拟合，提高泛化能力
4. **全栈集成**：从数据获取到链上交易的完整闭环

### 技术优势
1. **模块化设计**：各组件松耦合，易于维护和扩展
2. **异步架构**：充分利用 Python async/await 提高性能
3. **生产就绪**：包含完整的错误处理、日志记录、监控功能
4. **研究导向**：集成先进的深度学习技术

### 适用场景
- Solana meme 代币量化交易
- 高频策略研究和回测
- 自动化做市和套利
- 因子挖掘和策略发现

## 部署要求

### 硬件要求
- **训练阶段**：GPU（推荐 NVIDIA GPU，8GB+ 显存）
- **推理/交易阶段**：CPU 即可，稳定网络连接

### 软件要求
- **Python**：3.10+（推荐 3.11）
- **数据库**：PostgreSQL 12+（推荐 TimescaleDB 扩展）
- **Solana 节点**：QuickNode 或自有 RPC 节点

### 外部服务
1. **Birdeye API**：获取代币行情数据（需要 API 密钥）
2. **Solana RPC**：区块链交互（推荐 QuickNode）
3. **Jupiter API**：交易聚合（公开 API）

### 环境变量
创建 `.env` 文件包含：
```env
# 数据库
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crypto_quant

# API 密钥
BIRDEYE_API_KEY=your_birdeye_key
QUICKNODE_RPC_URL=https://your-quicknode-url

# 钱包
SOLANA_PRIVATE_KEY=your_base58_private_key
```

## 项目状态与注意事项

### 当前状态
根据 README 提示，项目目前处于"双方已达成和解"状态，但代码库完整且功能齐全。CATREADME.md 提供了详细的技术文档，可作为实际使用的指南。

### 注意事项
1. **实盘风险**：代码需要适当修改才能用于实盘交易，直接使用可能出现意外损失
2. **依赖管理**：缺少完整的依赖清单，需要根据错误提示逐步安装
3. **策略文件**：`best_meme_strategy.json` 需要先训练生成，仓库不包含预训练模型
4. **配置要求**：需要正确配置所有环境变量和外部服务

### 扩展建议
1. **添加更多数据源**：集成更多 DeFi 数据提供商
2. **多链支持**：扩展至 Ethereum、Avalanche 等其他链
3. **策略组合**：支持多个策略同时运行
4. **风险管理增强**：添加更复杂的风险控制规则
5. **性能优化**：优化数据管道和模型推理速度

## 结论

AlphaGPT 是一个专业级的量化交易系统，特别适合 Solana meme 代币的高频交易和策略研究。其创新的"生成公式+回测优化"方法为自动化策略发现提供了新思路，清晰的模块化架构使其具有良好的可维护性和扩展性。

对于量化交易研究人员和开发者，该项目提供了：
- 完整的量化交易系统实现参考
- 先进的深度学习技术在金融领域的应用案例
- 区块链与 AI 结合的实际工程示例
- 可扩展的框架用于自定义策略开发

尽管项目当前状态特殊，但其技术价值和工程实现仍具有很高的学习和参考价值。