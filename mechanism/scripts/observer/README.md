# 观测账号编排脚本

本目录存放机理建模专用的平台编排脚本。脚本遵循“主号直连、观测账号经代理”的隔离方式。

## 环境变量

默认读取两个 env 文件：

```text
# 仓库根目录 .env
LOGIN_KEY=...

# mechanism/scripts/observer/.env
OBSERVER_KEY=...
OBSERVER_PROXY=socks5h://127.0.0.1:1081
```

观测账号请求必须设置 `OBSERVER_PROXY`，脚本不会在代理缺失时回退到直连。

## 基本流程

先启动 SSH SOCKS 转发：

```bash
ssh -N goldrush-observer
```

只读检查：

```bash
conda run -n goldrush python mechanism/scripts/observer/submit_symmetry_probe_batch.py doctor
```

主号上传供他人挑战的公开 probe：

```bash
conda run -n goldrush python mechanism/scripts/observer/submit_symmetry_probe_batch.py publish-main --layout a --yes
```

观测账号经代理解析主号公开模型：

```bash
conda run -n goldrush python mechanism/scripts/observer/submit_symmetry_probe_batch.py resolve-main
```

观测账号经代理挑战主号公开模型，并在每局上传观测方 probe：

```bash
conda run -n goldrush python mechanism/scripts/observer/submit_symmetry_probe_batch.py run --layout a --map-id 1 --map-id 2 --count-per-map 5 --yes
```

下载双方视角 replay：

```bash
conda run -n goldrush python mechanism/scripts/observer/submit_symmetry_probe_batch.py fetch --run-id <run_id> --wait
```

## 安全边界

- `doctor`、`resolve-main`、`fetch` 不提交策略、不发起对局。
- `publish-main` 会调用 `add_model_4` 修改主号公开代码，必须显式 `--yes`。
- `run` 会调用 `add_model_1` 上传观测方代码并创建对局，必须显式 `--yes`。
- 脚本不会打印 key 或 token。
