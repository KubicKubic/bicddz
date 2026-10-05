# 斗地主 Bot API（`/api/v1/doudizhu`）

让程序（bot）以普通玩家身份参加 QOJ 斗地主：排队匹配、叫分、出牌、托管、聊天。接口直接调用网页使用的同一套服务端代码，所以 **规则、计时、超时托管、Rating 匹配、信息可见性都与真人玩家完全相同**，bot 看不到任何真人看不到的东西。

- 地址：`https://<站点>/api/v1/doudizhu/...`
- 格式：请求和响应都是 JSON（UTF-8）
- 认证：`Authorization: Bearer <密钥>`

---

## 目录

1. [快速开始](#1-快速开始)
2. [获取密钥与认证](#2-获取密钥与认证)
3. [通用约定](#3-通用约定)
4. [接口一览](#4-接口一览)
5. [大厅与匹配](#5-大厅与匹配)
6. [对局](#6-对局)
7. [对局状态对象](#7-对局状态对象)
8. [牌与牌型](#8-牌与牌型)
9. [规则、计分与计时](#9-规则计分与计时)
10. [错误](#10-错误)
11. [推荐的 bot 主循环](#11-推荐的-bot-主循环)
12. [完整示例](#12-完整示例)

---

## 1. 快速开始

```bash
KEY=YOUR_QOJ_API_KEY
BASE=https://qoj.ac/api/v1/doudizhu

# 排队（单局匹配）
curl -X POST "$BASE/queue" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"mode":"single"}'

# 轮询自己的状态，直到 game 不为 null（排队期间至少每 15 秒调用一次）
curl "$BASE/me" -H "Authorization: Bearer $KEY"

# 读取对局状态
curl "$BASE/games/123" -H "Authorization: Bearer $KEY"

# 轮到自己时叫分 / 出牌 / 不出（version 取自上一步读到的 state.version）
curl -X POST "$BASE/games/123/bid"  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"version":5,"value":0}'
curl -X POST "$BASE/games/123/play" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"version":9,"cards":[0,4,8,12,16]}'
curl -X POST "$BASE/games/123/pass" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"version":10}'
```

仓库中有可直接运行的示例：`scripts/doudizhu_bot_example.py`。

---

## 2. 获取密钥与认证

**密钥由管理员签发**，每个请求都带上：

```http
Authorization: Bearer qoj_...
```

密钥的权限（scope）是 `doudizhu:play`：

- 它只在 `/api/v1/doudizhu/*` 上生效。在其他 API 或网页上，它 **不算登录**。
- bot 账号就是普通账号。它的战绩、积分、Rating 与真人一样计算，也会出现在排行榜上。
- 以下情况会返回 `401`：账号被封禁、密钥已吊销、密钥错误或缺失。

---

## 3. 通用约定

| 项目 | 约定 |
|---|---|
| GET 参数 | 放在查询字符串，如 `?version=12&chat_after=3` |
| POST 参数 | 放在 JSON 对象请求体中（`Content-Type: application/json`）；请求体必须是 JSON 对象 |
| 类型 | 严格按 JSON 类型：`version`、`value` 等为整数，`on` 为布尔，`cards` 为整数数组 |
| 时间 | 所有时间戳均为 **Unix 毫秒**；`remaining` 等时长也以毫秒计 |
| 座位 | `seat` 为 0、1、2。出牌顺序为 0 → 1 → 2 → 0（下家 = `(seat + 1) % 3`） |
| 版本号 | 每局有一个递增的 `version`，状态每变化一次加 1（聊天不影响版本号） |
| 缓存 | 响应都带 `Cache-Control: no-store` |

### 版本号与乐观并发

每个改变对局的操作（叫分、出牌、不出、托管）都 **必须带上它所依据的 `version`**：

- 版本号与服务器一致、且操作合法时，操作生效，返回 `200` 和新状态。
- 版本号已过期（牌局已经变了）、还没轮到你或操作不合法时，返回 `409`，附带错误信息和 **当前最新状态**，操作不生效。

这样可以保证重试、网络重放或多个进程并发时，不会把旧决策作用到新局面上。收到 `409` 后，用返回的 `state` 重新决策即可。

---

## 4. 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | [`/info`](#get-info) | 常量：牌编码、牌型名、计时、比赛局数 |
| GET | [`/me`](#get-me) | 自己的状态（当前对局、排队、Rating）；**同时是排队心跳** |
| GET | `/lobby` | 同 `/me` |
| POST | [`/queue`](#post-queue) | 排队：单局匹配或记分比赛 |
| POST | [`/cancel`](#post-cancel) | 退出队列 |
| GET | [`/games/{id}`](#get-gamesid) | 对局状态（可增量） |
| GET | [`/games/{id}/hints`](#get-gamesidhints) | 当前可出的牌组合 |
| POST | [`/games/{id}/bid`](#post-gamesidbid) | 叫分 |
| POST | [`/games/{id}/play`](#post-gamesidplay) | 出牌 |
| POST | [`/games/{id}/pass`](#post-gamesidpass) | 不出 |
| POST | [`/games/{id}/auto`](#post-gamesidauto) | 开关托管 |
| POST | [`/games/{id}/chat`](#post-gamesidchat) | 发言 |

---

## 5. 大厅与匹配

### GET /info

返回常量，便于 bot 自检：

```json
{
  "cards": "id 0..51 = rank * 4 + suit (suit 0 ♠, 1 ♥, 2 ♣, 3 ♦); 52 small joker, 53 big joker",
  "ranks": ["3","4","5","6","7","8","9","10","J","Q","K","A","2","小王","大王"],
  "pattern_types": {"rocket":"王炸","bomb":"炸弹","single":"单张","pair":"对子","trio":"三张",
                    "straight":"顺子","pairs":"连对","plane":"飞机","trio1":"三带一","trio2":"三带一对",
                    "four2":"四带二","four22":"四带两对","plane1":"飞机带单","plane2":"飞机带对"},
  "bid_seconds": 15, "play_seconds": 15, "reserve_seconds": 15,
  "queue_stale_seconds": 15, "match_rounds": 6
}
```

### GET /me

返回自己的状态。**这个接口同时是排队心跳，并会触发匹配**：排队期间如果超过 15 秒没有调用，就会被移出队列（与网页关闭页面相同）。

```json
{
  "username": "ddz-bot-1",
  "game": 123,
  "queued": null,
  "queued_ms": null,
  "rating": 1500,
  "window": null,
  "queue_size": {"single": 2, "match": 0}
}
```

| 字段 | 说明 |
|---|---|
| `game` | 正在进行的对局 id；没有则为 `null`。**匹配成功后这里出现对局 id** |
| `queued` | 正在排的队：`"single"`、`"match"` 或 `null` |
| `queued_ms` | 已排队的毫秒数 |
| `rating` | 自己的斗地主 Rating（来自记分比赛；没有记录时为 1500） |
| `window` | 当前可接受的同桌 Rating 差（三人最高减最低），50～200；不在排队时为 `null`。见 [匹配规则](#匹配规则) |
| `queue_size` | 两个队列中的有效人数 |

### POST /queue

```json
{"mode": "single"}
```

- `mode`：`"single"`（单局匹配，打一局）或 `"match"`（记分比赛，同三人连打 6 局，计算 Rating）。省略时为 `"single"`。
- 一个账号同时只能在一个队列中；换模式会自动离开原队列。
- 已经在对局中时不会排队，直接返回带 `game` 的状态。
- 返回值与 `GET /me` 相同。

### POST /cancel

退出队列。请求体可以为空，返回值与 `GET /me` 相同。

### 匹配规则

- 两个模式都按 Rating 匹配，没有 Rating 时按 1500。
- **可接受的同桌 Rating 差**：等待者可以接受的三人 Rating 差（最高减最低）为 `50 + 已等待秒数`，最多 200（等满 150 秒达到上限）。相差超过 200 的玩家永远不会同桌。
- **选人顺序**：优先照顾等得最久的人，在他的可接受范围内，选 Rating 差最小的另外两人。
- **座位与首叫**：座位顺序随机，首个叫分者也随机。

---

## 6. 对局

### GET /games/{id}

读取对局状态。

| 查询参数 | 说明 |
|---|---|
| `version`（可选） | 上次读到的版本号。版本没变且没有时钟到期时，返回精简响应（见下） |
| `chat_after`（可选） | 聊天消息 id；返回该 id 之后的聊天（从头取用 `0`） |

完整响应：

```json
{"state": { ...对局状态对象... }, "chat": [ ... ]}
```

带 `version` 且没有变化时，返回精简响应：

```json
{"state": {"unchanged": true, "version": 17, "remaining": 12345}}
```

`chat` 只在请求带 `chat_after` 时出现，每条消息的格式为：

```json
{"id": 8, "seat": 1, "username": "alice", "t": 1790000000000, "text": "炸得好！"}
```

**谁能读**：进行中的对局只有本局玩家能读；已结束的对局任何人都能读，此时会公开全部手牌和洗牌记录。

### GET /games/{id}/hints

轮到自己出牌时，返回可以打出的牌组合，每个组合是一个牌 id 数组。**从弱到强排列，炸弹和王炸在最后**；自己是本轮首家时，按点数从小到大每个点数给一组。不是自己出牌时（包括叫分阶段）返回空数组。

```json
{"hints": [[4, 5], [36, 37], [8, 9, 10, 11], [52, 53]]}
```

提示只列出一部分合理组合，并不穷举所有合法出法。bot 可以打出任何合法组合，不必只从提示里选。

### POST /games/{id}/bid

```json
{"version": 5, "value": 2}
```

- `value` 可以是 `0`（不叫）、`1`、`2` 或 `3`。
- 非 0 的叫分必须高于当前最高分（`state.bid`）。
- `state.must_bid` 为 `true` 时不能叫 0，见 [叫分](#叫分)。

### POST /games/{id}/play

```json
{"version": 9, "cards": [0, 4, 8, 12, 16]}
```

- `cards`：要打出的牌 id，1～20 张，必须都在自己手里，顺序无所谓。
- `choice`（可选）：一手牌有多种合法理解时，用来指定按哪种理解打出。格式为 `"type:rank:len"`，例如 `"plane1:3:3"`。
  - 不指定时，首家出牌按「牌型优先级、点数高者优先」取第一种。
  - 跟牌时，取第一种能管住上家的理解。

  例：`333444555666` 可以是 `plane:3:4`（四连飞机），也可以是 `plane1:3:3`（飞机 456 带 3、3、3）或 `plane1:2:3`（飞机 345 带 6、6、6）。

### POST /games/{id}/pass

```json
{"version": 10}
```

自己是本轮首家时不能不出，此时返回 `409`。

### POST /games/{id}/auto

```json
{"version": 11, "on": false}
```

开启或关闭托管。**超时会自动进入托管**，要继续自己出牌必须先发送 `{"on": false}`。托管规则见 [计时与托管](#计时与托管)。

### POST /games/{id}/chat

```json
{"text": "大家好"}
```

- 只有本局玩家可以发言，对局进行中和结束后都可以，不需要 `version`。
- 每条最多 60 字；每个账号每 2 秒最多一条，发得太快返回 `403`。
- 可以带 `chat_after`，一并取回新消息。
- 返回 `{"ok": true}`，带 `chat_after` 时附带 `chat`。

### 成功响应（bid / play / pass / auto）

```json
{"ok": true, "state": { ...新的对局状态... }}
```

---

## 7. 对局状态对象

下表是 `state` 的字段。标 ★ 的字段是 bot 决策时最常用的。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int | 对局 id |
| ★`version` | int | 状态版本号，操作时原样回传 |
| ★`phase` | string | `"bidding"`（叫分）、`"playing"`（出牌）或 `"finished"`（已结束） |
| ★`seat` | int\|null | 自己的座位；观战者为 `null` |
| ★`turn` | int\|null | 当前该谁行动；已结束时为 `null` |
| ★`remaining` | int\|null | 距当前行动者超时的毫秒数（包含其剩余局时） |
| `reserve_in_remaining` | int | `remaining` 中属于局时的部分（毫秒）：`remaining - reserve_in_remaining` 就是本步剩余时间 |
| `reserve` | int[3] | 每人剩余局时（毫秒） |
| ★`hand` | int[]\|null | 自己的手牌（牌 id，升序）；观战者为 `null` |
| `hands` | int[3][]\|null | 所有人的手牌，只在对局结束后公开 |
| `players` | object[3] | 每个座位的信息，见下表 |
| ★`bid` | int | 当前最高叫分（地主确定后就是底分） |
| ★`must_bid` | bool | 为 `true` 时轮到的人不能叫 0 |
| ★`landlord` | int\|null | 地主座位；叫分阶段为 `null` |
| `bottom` | int[3]\|null | 底牌；地主确定后公开 |
| ★`last` | object\|null | 当前要压的牌：`{"seat": 座位, "pattern": 牌型}`；这一轮还没人出牌时为 `null` |
| ★`leading` | bool | 自己（当前行动者）是否是本轮首家。为 `true` 时可以出任意牌型，并且不能不出 |
| `table` | array[3] | 桌面上每个座位最近一次动作：`null`、`"pass"` 或 `{"cards": [...], "pattern": {...}}` |
| `multiplier` | int | 当前倍数 |
| `bombs` | int | 已出的炸弹和王炸数 |
| `redeals` | int | 本轮叫分前已连续流局的次数 |
| `log` | object[] | 全部出牌记录，见下文 |
| `result` | object\|null | 结算结果，对局结束后才有，见下文 |
| `started` | int | 开局时间（毫秒） |
| `match` | object\|null | 记分比赛信息；单局时为 `null`，见下文 |
| `fairness` | object | 洗牌承诺 `commitments`（每次发牌一个 SHA-256）；对局结束后另有 `deals`（牌序 `deck`、随机盐 `salt`、首叫 `first_bidder`），可自行核验 |

### players[i]

| 字段 | 说明 |
|---|---|
| `username` | 用户名 |
| `count` | 剩余张数 |
| `auto` | 是否托管中 |
| `bid` | 叫分阶段该座位本轮的叫分：`null` 表示还没叫，`0` 表示不叫，`1`～`3` 为叫分；出牌阶段固定为 `null` |
| `role` | `"landlord"`（地主）、`"farmer"`（农民）；叫分阶段为 `null` |

### log[]

每条记录都有 `t`（毫秒时间戳）和 `kind`，其余字段因 `kind` 而异：

| `kind` | 其他字段 | 含义 |
|---|---|---|
| `bid` | `seat`, `value`, `auto` | 叫分（0 表示不叫） |
| `redeal` | — | 三家都不叫，重新洗牌发牌 |
| `landlord` | `seat`, `value`, `cards` | 成为地主，叫分为 `value`，`cards` 为底牌 |
| `play` | `seat`, `cards`, `pattern`, `auto` | 出牌 |
| `pass` | `seat`, `auto` | 不出 |
| `auto_on` | `seat`, `timeout?` | 进入托管（`timeout: true` 表示因超时进入） |
| `auto_off` | `seat` | 取消托管 |
| `finish` | `seat` | 该座位出完所有牌 |

`auto: true` 表示这一步是托管自动完成的。

### result

```json
{"winner": 0, "landlord_won": true, "spring": false, "anti_spring": false,
 "base": 3, "multiplier": 2, "deltas": [12, -6, -6], "finished": 1790000000000}
```

`deltas[i]` 是各座位本局的得分，三人之和为 0。

### match（记分比赛）

| 字段 | 说明 |
|---|---|
| `id` | 比赛 id |
| `round` / `rounds` | 本局是第几局 / 共几局（当前为 6） |
| `players` | 三人（按座位） |
| `games` | 已创建的各局对局 id |
| `deltas` | 各局得分 `[[d0,d1,d2], ...]` |
| `totals` | 目前总分 |
| `next_game` | 本局结束后的下一局 id；比赛结束或下一局还没创建时为 `null` |
| `finished` | 比赛是否已结束 |
| `places`, `rating_before`, `rating_after` | 比赛结束后的名次和 Rating 变化 |

**一局结束后，下一局会立即创建**，发牌延后 5 秒。此时 `GET /me` 的 `game` 会指向新的对局。

---

## 8. 牌与牌型

### 牌 id

| id | 牌 |
|---|---|
| `0`～`51` | `点数 × 4 + 花色` |
| `52` | 小王 |
| `53` | 大王 |

- **点数（rank）**：`0`=3, `1`=4, `2`=5, `3`=6, `4`=7, `5`=8, `6`=9, `7`=10, `8`=J, `9`=Q, `10`=K, `11`=A, `12`=2, `13`=小王, `14`=大王。
- **花色**：`0`=♠, `1`=♥, `2`=♣, `3`=♦。花色不影响大小。
- 计算方式：`rank = id < 52 ? id // 4 : id - 39`。

例：♠3 = 0，♥3 = 1，♠A = 44，♦2 = 51，大王 = 53。

### 牌型对象 `pattern`

```json
{"type": "straight", "rank": 7, "len": 6}
```

- `type`：牌型，见下表。
- `rank`：比较大小用的点数。对顺子、连对、飞机类牌型，是主体中 **最大** 的点数；对带牌的牌型，是主体（三张、四张）的点数。
- `len`：顺子类牌型是主体的单位数（几张单牌、几对、几组三张）；其他牌型为 1。

| `type` | 名称 | 构成 |
|---|---|---|
| `single` | 单张 | 1 张 |
| `pair` | 对子 | 2 张同点 |
| `trio` | 三张 | 3 张同点 |
| `trio1` | 三带一 | 三张 + 1 张单牌 |
| `trio2` | 三带一对 | 三张 + 1 对 |
| `straight` | 顺子 | ≥5 张连续单牌 |
| `pairs` | 连对 | ≥3 对连续对子 |
| `plane` | 飞机 | ≥2 组连续三张 |
| `plane1` | 飞机带单 | n 组连续三张 + n 张单牌 |
| `plane2` | 飞机带对 | n 组连续三张 + n 对 |
| `four2` | 四带二 | 四张 + 2 张单牌（可以是一对） |
| `four22` | 四带两对 | 四张 + 2 对 |
| `bomb` | 炸弹 | 4 张同点 |
| `rocket` | 王炸 | 大王 + 小王 |

**构成限制**：

- 顺子、连对、飞机的主体只能是 3～A，不能包含 2 和王。
- 带的牌不能和主体同点，也不能包含炸弹（4 张同点）或王炸。
- 例：`33334444` 不是合法牌型；`3333` + 大小王 也不是四带二。

**比较规则**：

- 同牌型、同 `len` 时，比较 `rank`。
- 炸弹可以管住任何非炸弹牌型；大炸弹管小炸弹。
- 王炸最大。
- 其他情况都管不上。

`choice` 使用的牌型键就是 `"type:rank:len"`。

---

## 9. 规则、计分与计时

### 发牌

- 一副 54 张，每人 17 张，留 3 张底牌。
- 服务端用密码学安全随机数做 Fisher–Yates 洗牌。
- 发牌前公布牌序的承诺（`fairness.commitments`），对局结束后公开完整牌序（`fairness.deals`），可自行核验。

### 叫分

- 从随机选出的首叫者开始，按座位顺序每人叫一次。
- 可以叫 0（不叫），或叫比当前最高分更高的 1/2/3 分。
- 叫 3 分立即成为地主；否则三人都叫过后，叫分最高者成为地主。
- 地主获得底牌，并首先出牌。
- 三人都不叫时重新洗牌发牌（记为 `redeal`）。
- 连续流局 3 次后，下一轮的最后一位叫分者如果前两人都没叫，就必须叫分，此时 `must_bid` 为 `true`。

### 出牌

- 按座位顺序轮流行动。
- 跟牌必须出能管住 `last` 的牌，或者不出。
- 其余两家都不出时，最后出牌的人重新自由出牌，此时 `leading` 为 `true`，不能不出。
- 先出完手牌的一方获胜：地主出完则地主胜，任一农民出完则农民胜。

### 计分

- **底分**：地主叫的分。
- **倍数**：从 1 开始，以下情况各 **+1**：
  - 每出一个炸弹或王炸；
  - 春天：地主获胜，且两个农民一张牌都没出；
  - 反春：农民获胜，且地主只出过第一手牌。

  例：一个炸弹 ×2，两个炸弹 ×3，春天加两个炸弹 ×4。
- **得分**：地主获胜时，地主 +2 × 底分 × 倍数，每个农民 −底分 × 倍数；农民获胜时正负相反。

### 记分比赛与 Rating

- 同三人连打 6 局，每人从 0 分开始累计，打完按总分排名。
- Rating 按 Elo 计算：初始 1500，每两人按总分比较（高者算胜，相同算平），K = 32。
- 单局也计入积分榜，但不影响 Rating。

### 计时与托管

| 阶段 | 时限 |
|---|---|
| 叫分 | 每次 15 秒 |
| 出牌 | 每步 15 秒；另有每人整局 15 秒的 **局时**，某一步超过 15 秒的部分从局时扣除 |

- **超时**：服务器替你行动，并让你 **进入托管**（`players[seat].auto = true`）。
- **托管中**：服务器每约 1.2 秒替你行动一次。
  - 叫分：按手牌强弱自动叫分。
  - 出牌：手牌能一次出完就直接出完；否则首家出最小的一组，跟牌时出能管上的最小非炸弹组合；不压队友。
- **取消托管**：发送 `POST /auto {"on": false}` 取回控制权，时钟重新开始计时。

**注意**：你的操作如果恰好发生在服务器替你执行超时动作的同一请求里，会返回 `409` 和 `"操作超时，已自动托管。"`。这时你已经在托管中，需要先取消托管。

---

## 10. 错误

错误响应的格式为：

```json
{"error": "说明文字"}
```

`409` 的响应还会附带 `state`。

| 状态码 | 含义 | 处理 |
|---|---|---|
| 400 | 参数格式错误（缺少 `version`、`value` 不是 0～3 的整数、`cards` 不是整数数组等） | 修正请求 |
| 401 | 没有有效密钥（缺失、错误、已吊销，或账号被封禁） | 检查密钥 |
| 403 | 无权操作：不是本局玩家、不能查看进行中的对局、聊天太快、聊天暂时不可用等 | 按信息处理 |
| 404 | 对局不存在，或接口路径不存在 | — |
| 405 | 方法不对（例如用 GET 调用 `/queue`） | 改用正确的方法 |
| 409 | 操作被拒绝，附带当前 `state`（见下表） | 用返回的 `state` 重新决策 |
| 503 | 服务器暂时繁忙 | 稍后重试 |

`409` 的 `error` 可能是：

| error | 原因 |
|---|---|
| `牌局已更新，请重新操作。` | `version` 已过期 |
| `操作超时，已自动托管。` | 时钟已到期，服务器已替你行动并开启托管 |
| `还没有轮到你叫分。` / `还没有轮到你出牌。` | 不是你的回合，或阶段不对 |
| `叫分必须高于当前分数。` | 叫分没有高于 `bid` |
| `连续流局，本轮你必须叫分。` | `must_bid` 时叫了 0 |
| `只能打出自己手里的牌。` | `cards` 中有不在手里的牌或重复的牌 |
| `这不是合法的牌型。` | 不构成任何牌型 |
| `所选的牌管不上。` | 牌型合法，但管不住 `last` |
| `你是本轮首家，必须出牌。` | 首家不能不出 |

---

## 11. 推荐的 bot 主循环

```
POST /queue {"mode": ...}
loop:
    me = GET /me                      # 排队心跳 + 触发匹配；至少每 15 秒一次
    if me.game is null: sleep 1s; continue
    version = null
    loop:
        r = GET /games/{me.game}?version={version}
        if r.state.unchanged: sleep 0.3s; continue
        s = r.state; version = s.version
        if s.phase == "finished": break           # 记分比赛会接着开下一局：回到外层循环读 /me
        if s.turn != s.seat: sleep 0.3s; continue
        if s.players[s.seat].auto: POST /auto {"version": version, "on": false}; continue
        决策并 POST bid / play / pass（带上 version）
        # 200：新状态已在响应里；409：用响应里的 state 重新决策
```

建议：

- 对局中每 0.3～1 秒轮询一次。带 `version` 的请求在没有变化时开销很小。
- 叫分每次只有 15 秒，出牌每步 15 秒外加整局 15 秒局时，决策要在这个时间内完成。
- 不要并发发送同一局的多个操作；即使发了，也只有基于最新 `version` 的那个会生效。

---

## 12. 完整示例

一个最简单的 Python bot：永不主动叫分，按提示出最小的牌，不压队友。

```python
import http.client, json, sys, time
from urllib.parse import urlsplit

BASE, KEY, MODE = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "single")
URL = urlsplit(BASE)

def api(method, path, body=None):
    Conn = http.client.HTTPSConnection if URL.scheme == "https" else http.client.HTTPConnection
    conn = Conn(URL.netloc, timeout=30)
    conn.request(method, "/api/v1/doudizhu" + path, None if body is None else json.dumps(body),
                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"{}")

def play(game):
    version = None
    while True:
        _, data = api("GET", f"/games/{game}" + (f"?version={version}" if version else ""))
        s = data["state"]
        if s.get("unchanged"):
            time.sleep(0.3); continue
        version = s["version"]
        if s["phase"] == "finished":
            return s["result"]["deltas"][s["seat"]]
        if s["turn"] != s["seat"]:
            time.sleep(0.3); continue
        if s["players"][s["seat"]]["auto"]:
            api("POST", f"/games/{game}/auto", {"version": version, "on": False}); continue
        if s["phase"] == "bidding":
            value = max(1, s["bid"] + 1) if s["must_bid"] else 0
            api("POST", f"/games/{game}/bid", {"version": version, "value": value}); continue
        partner = (not s["leading"] and s["landlord"] != s["seat"] and s["last"]["seat"] != s["landlord"])
        hints = api("GET", f"/games/{game}/hints")[1]["hints"]
        if hints and not partner:
            api("POST", f"/games/{game}/play", {"version": version, "cards": hints[0]})
        elif not s["leading"]:
            api("POST", f"/games/{game}/pass", {"version": version})

api("POST", "/queue", {"mode": MODE})
while True:
    _, me = api("GET", "/me")
    if not me.get("game"):
        if not me.get("queued"):
            api("POST", "/queue", {"mode": MODE})
        time.sleep(1); continue
    print(me["username"], "game", me["game"], "score", play(me["game"]))
```

运行：`python3 bot.py https://qoj.ac qoj_xxxx single`。
