# GoldRush2.0 官方评测平台接口探测

本文记录官方评测平台及其自动化相关探测结论。不得写入 `.env` 中的 key、登录返回 token 或其他敏感凭据。

## 基本信息

- 平台地址：`http://47.103.127.219/`
- 前端形态：Vue 单页应用。
- 登录页入口资源：
  - `/assets/index-D8Y_F_zB.js`
  - `/assets/index-vGbtvkBv.js`

## 登录

### `POST /api/user/login`

请求体：

```json
{
  "key": "<LOGIN_KEY>"
}
```

响应结构：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "token": "<token>",
    "user": {
      "id": 142,
      "name_cn": "gary318",
      "name_en": "Ubiquant142",
      "role": 1,
      "is_ban": 0,
      "school": "清华大学"
    }
  },
  "timestamp": 1784874019174
}
```

前端行为：

- 登录成功后，前端将 `data.token` 写入 `localStorage.token`。
- `role=1` 跳转 `/dashboard/`。
- `role=2` 跳转 `/admin/`。

## 鉴权

前端 Axios 请求拦截器会将 token 原样写入请求头：

```http
Authorization: <token>
```

注意：当前前端没有添加 `Bearer ` 前缀。

## 用户信息

### `GET /api/user/get_user_info`

请求头：

```http
Authorization: <token>
```

已验证该接口可用于确认登录会话有效。

## Dashboard

### 页面资源

- 页面地址：`/dashboard/`
- 主入口：`/assets/dashboard-C542ejVt.js`
- 依赖 chunk：
  - `/assets/index-vGbtvkBv.js`
  - `/assets/el-tag-C6FH1uxN.js`
  - `/assets/el-button-fnb_VpS4.js`
  - `/assets/el-select-Bw8tGrSb.js`
  - `/assets/popup-ROvw4pYX.js`

Dashboard 页面要求用户角色为 `role=1`。

### 当前阶段

### `GET /api/user/get_stage`

用途：获取当前赛事阶段，Dashboard 按阶段显示不同入口。

响应字段：

```json
{
  "stage": 1
}
```

前端阶段逻辑：

- `stage=1`：显示公测相关入口，包括“上传代码”和“上传供他人比赛代码”。
- `stage=2` 或 `stage=3`：显示“参与比赛”入口。
- `stage=1`、`stage=3`，或 `stage=2` 且存在初赛数据时，展示 Dashboard 主内容。

### 用户信息扩展字段

### `GET /api/user/get_user_info`

Dashboard 上传弹窗会额外读取：

- `today_initiated`：今日已提交次数。
- `daily_initiate_limit`：今日提交次数上限。

当前验证值：

```json
{
  "today_initiated": 1,
  "daily_initiate_limit": 300
}
```

### 地图列表

### `GET /api/user/get_map_list`

用途：

- 公测记录筛选地图。
- 公测上传时选择地图。

响应字段：

```json
{
  "list": [
    {
      "id": 2,
      "user_id": 1,
      "name": "地图2",
      "is_ban": 0,
      "created_at": "2026-07-17T04:34:30.000Z",
      "updated_at": "2026-07-20T06:31:21.000Z"
    }
  ]
}
```

当前验证返回 2 张地图。

### 公测游戏记录

### `GET /api/user/get_game_list_1`

用途：

- Dashboard 首页展示最近公测记录。
- “查看更多”弹窗展示完整公测记录。

首页请求参数：

```json
{
  "page": 1,
  "page_size": 6
}
```

弹窗请求参数：

```json
{
  "page_size": 1000,
  "map_id": "<可选>",
  "start_date": "<可选>",
  "end_date": "<可选>"
}
```

响应字段：

```json
{
  "list": [
    {
      "id": 22096,
      "stage": 1,
      "user_id": 142,
      "user_id2": 142,
      "map_id": 1,
      "is_upload_log": 1,
      "is_parse_log": 1,
      "error_msg": "",
      "created_at": "2026-07-24T05:29:52.000Z",
      "updated_at": "2026-07-24T05:30:00.000Z",
      "map_name": "地图1",
      "players": [
        {
          "game_id": 22096,
          "is_win": 1,
          "coin_num": 72,
          "model_name": "player1",
          "user_name_cn": "gary318"
        }
      ]
    }
  ],
  "total": 1
}
```

前端行为：

- `is_upload_log=2` 显示“游戏异常”。
- `is_parse_log=true` 时显示“回放”，点击跳转 `/game/?id=<game id>`。
- 否则显示“游戏中”。
- 弹窗中会对 `players[].coin_num` 取最大值作为“最高获得”。

### 初赛游戏记录

### `GET /api/user/get_game_list_2`

用途：

- Dashboard 首页展示最近初赛记录。
- “查看更多”弹窗展示完整初赛记录。

首页请求参数：

```json
{
  "page": 1,
  "page_size": 3
}
```

弹窗请求参数：

```json
{
  "page_size": 1000,
  "start_date": "<可选>",
  "end_date": "<可选>"
}
```

响应字段：

```json
{
  "list": [
    {
      "u1_name_cn": "<玩家1>",
      "u2_name_cn": "<玩家2>",
      "is_win": true,
      "created_at": "<时间>",
      "game_id": 123
    }
  ],
  "total": 0
}
```

当前验证账号暂无初赛记录，`list=[]`。

### 初赛统计

### `GET /api/user/get_game_data_2`

用途：展示初赛战绩统计。

响应字段：

```json
{
  "total_count": 0,
  "win_count": 0,
  "max_coin_num": 0,
  "rank_no": 0,
  "win_rate": "0%",
  "cost2": 0
}
```

### 可挑战模型列表

### `GET /api/user/get_model_list_4`

用途：公测上传时选择“他人代码”作为对手。

响应字段：

```json
{
  "list": [
    {
      "id": 17453,
      "stage": 4,
      "user_id": 62,
      "lang": 1,
      "name": "player62",
      "created_at": "2026-07-23T06:14:57.000Z",
      "updated_at": "2026-07-24T04:47:03.000Z",
      "user_name_cn": "wyfff"
    }
  ]
}
```

当前验证返回 26 个模型。

### 公测提交并发起对局

### `POST /api/user/add_model_1`

用途：公测阶段上传代码并发起对局。

请求类型：`multipart/form-data`

字段：

- `map_id`：地图 ID。
- `model_files`：代码文件，可重复。
- `model_langs`：语言，可重复。`1=Python`，`2=C++`。
- `model_names`：模型名，可重复；需要字母开头，且仅包含字母和数字。
- `model_id`：选择他人代码作为对手时提供。

前端模式：

- 自己两份代码对战：上传两个 `model_files` / `model_langs` / `model_names`。
- 选择他人代码：上传自己一份代码，并额外传 `model_id`。

文件类型：

- Python：`.py`
- C++：`.so`

参数约束与返回判读：

- `model_names` 不接受下划线。实测 `probe_exc_20260724_01`、`probe_stream_exc_20260724_01`、`probe_raw_return_20260724_01` 均被拒绝。
- 上述失败响应仍为 HTTP `200`，业务返回 `code=1`，`message="Model名称仅限字母和数字, 字母开头"`，且 `data.game_id` 为空。
- 改用较短的字母数字名称可成功，例如 `ProbeA01`、`ProbeB01`、`ProbeC01`、`Norm01`。
- 模型名过长时也可能返回同一条校验提示；为减少变量，自动化探测中建议使用短字母数字名。
- 上传成功以业务字段 `code=0` 和 `data.game_id` 存在为准，不应只看 HTTP 状态。
- `code=0` 只表示已创建对局，不表示模型运行正常。异常判负或格式非法仍可能生成 `is_upload_log=1`、`is_parse_log=1` 的可解析回放，并在 `get_game_log` 末尾写入 `forfeit`。
- 提交后建议轮询 `GET /api/user/get_game_info?id=<game_id>`，直到 `is_upload_log` 为 `1` 或 `2`；若 `is_parse_log=1`，再请求 `get_game_log` 检查是否存在 `forfeit`。

实测记录：

- 测试时间：2026-07-24。
- 上传文件：`official_sdk/code/player.py`，大小 2222 bytes。
- 地图：`map_id=2`，`地图2`。
- 上传模型：
  - `sdkplayerA`，`model_lang=1`
  - `sdkplayerB`，`model_lang=1`
- HTTP 状态：`200`
- 接口响应：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "game_id": 22240
  },
  "timestamp": 1784874582975
}
```

随后通过 `GET /api/user/get_game_list_1?page=1&page_size=5` 确认对局已生成并解析：

```json
{
  "id": 22240,
  "stage": 1,
  "user_id": 142,
  "user_id2": 142,
  "map_id": 2,
  "is_upload_log": 1,
  "is_parse_log": 1,
  "error_msg": "",
  "created_at": "2026-07-24T06:29:42.000Z",
  "updated_at": "2026-07-24T06:29:50.000Z",
  "map_name": "地图2",
  "players": [
    {
      "game_id": 22240,
      "is_win": 1,
      "coin_num": -56,
      "model_name": "sdkplayerA",
      "user_name_cn": "gary318"
    },
    {
      "game_id": 22240,
      "is_win": 0,
      "coin_num": -71,
      "model_name": "sdkplayerB",
      "user_name_cn": "gary318"
    }
  ]
}
```

### 初赛/比赛阶段提交

### `POST /api/user/add_model_2`

用途：`stage=2` 或 `stage=3` 时参与比赛。

请求类型：`multipart/form-data`

字段：

- `model_file`：代码文件。
- `model_lang`：语言。`1=Python`，`2=C++`。

前端模型名显示为系统分配：`player<user.id>`。

### 上传供他人比赛代码

### `POST /api/user/add_model_4`

用途：上传供其他选手在公测阶段选择挑战的代码。

请求类型：`multipart/form-data`

字段：

- `model_file`：代码文件。
- `model_lang`：语言。`1=Python`，`2=C++`。

前端模型名显示为系统分配：`player<user.id>`。

### Dashboard 外链

- 点击回放跳转 `/game/?id=<game id>`。
- 点击排行榜跳转 `/rank/`。

## Game 回放页

### 页面资源

- 页面地址：`/game/?id=<game_id>`
- 已探测样本：`/game/?id=22240`
- 主入口：`/assets/game-CU-ie88H.js`
- 额外脚本：`/assets/createjs.js`
- 依赖 chunk：
  - `/assets/index-vGbtvkBv.js`
  - `/assets/el-tag-C6FH1uxN.js`
  - `/assets/el-select-Bw8tGrSb.js`
- 样式：
  - `/assets/index-Yq6U_Lkg.css`
  - `/assets/el-select-yDJtNqOC.css`
  - `/assets/el-tag-DJFMEN6_.css`
  - `/assets/game-77EMSFzg.css`
- 图像素材：回放加载 `/assets/game/*.png`，包括角色方向、NPC、炸弹、地面、金币、障碍等素材。

页面允许 `role=1` 和 `role=2` 访问。管理员 `role=2` 会额外显示播放速度选择和指标面板。

### 对局信息

### `GET /api/user/get_game_info`

请求参数：

```json
{
  "id": 22240
}
```

用途：

- 获取对局元信息。
- 判断是否可播放回放。
- 获取当前用户视角 `viewer_side`。

样本响应：

```json
{
  "id": 22240,
  "stage": 1,
  "user_id": 142,
  "user_id2": 142,
  "map_id": 2,
  "is_upload_log": 1,
  "is_parse_log": 1,
  "error_msg": "",
  "created_at": "2026-07-24T06:29:42.000Z",
  "updated_at": "2026-07-24T06:29:50.000Z",
  "players": [
    {
      "model_name": "sdkplayerA",
      "is_win": 1,
      "user_name_cn": "gary318"
    },
    {
      "model_name": "sdkplayerB",
      "is_win": 0,
      "user_name_cn": "gary318"
    }
  ],
  "viewer_side": 1,
  "is_admin": false
}
```

前端行为：

- `is_upload_log=2`：显示“对局异常”，使用 `error_msg` 作为错误信息，不请求日志。
- `is_parse_log` 为假：提示“游戏中, 无法回放”。
- 否则继续请求 `get_game_log`。

补充实测：

- 选手代码运行异常或返回格式非法时，不一定表现为 `is_upload_log=2`。
- `game_id=22373`、`22375`、`22376` 均为 `is_upload_log=1`、`is_parse_log=1`、`error_msg=""`，但 `get_game_log` 末尾存在 `forfeit` 行记录判负原因和部分详情。
- 因此判断选手判负详情时，应同时检查 `get_game_info.error_msg` 和 `get_game_log` 中的 `forfeit`。

### 对局日志

### `GET /api/user/get_game_log`

请求参数：

```json
{
  "id": 22240
}
```

响应类型：`application/x-ndjson; charset=utf-8`。

返回体为回放日志。具体格式、字段含义和样本分析见 `gamerules/replay.md`。
