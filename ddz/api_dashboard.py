"""Read-only QOJ terminal / HTML views; no inference or HTTP on the UI thread."""
import html
import json
import time
import unicodedata

RANKS=('3','4','5','6','7','8','9','10','J','Q','K','A','2','小王','大王')
PHASES={'bidding':'叫分','playing':'出牌','finished':'已结束'}

def clean(value):
    return ''.join(c for c in str(value) if not unicodedata.category(c).startswith('C'))

def clip(text,width):
    out=''; used=0
    for char in clean(text):
        size=2 if unicodedata.east_asian_width(char) in ('W','F') else 1
        if used+size>width: break
        out+=char; used+=size
    return out

def cards(ids):
    # Ranks are sufficient for the view; real physical IDs remain in JSON.
    return ' '.join(RANKS[c//4 if c<52 else c-39] for c in sorted(ids or ())
                    if type(c) is int and 0<=c<54) or '—'

def role(seat,landlord):
    if landlord is None:return '未定'
    return ('地主','下家','门板')[(seat-landlord)%3]

def public_snapshot(raw):
    """Never include finished-game opponent hands, fairness decks or chat."""
    fields=('id','version','phase','seat','turn','remaining','reserve','hand','bottom',
            'bid','must_bid','landlord','last','leading','table','multiplier','bombs',
            'redeals','log','result','match')
    result={k:raw[k] for k in fields if k in raw}
    result['players']=[{k:p[k] for k in ('username','count','auto','bid','role') if k in p}
                       for p in raw.get('players',())]
    result['observed_at']=time.time()
    return result

def action_text(endpoint,payload):
    if endpoint=='bid':return f"叫 {payload.get('value')} 分"
    if endpoint=='pass':return '不出'
    if endpoint=='play':return '出 '+cards(payload.get('cards'))+'  '+str(payload.get('choice','合法提示'))
    return str(endpoint or '等待')

def event_text(entry):
    actor=f"S{entry.get('seat','?')}"
    kind=entry.get('kind')
    if kind=='play':msg='出 '+cards(entry.get('cards'))
    elif kind=='bid':msg=f"叫 {entry.get('value')} 分"
    elif kind=='pass':msg='不出'
    elif kind=='landlord':msg='成为地主，底牌 '+cards(entry.get('cards'))
    elif kind=='redeal':msg='全不叫，重新发牌'
    else:msg=str(kind)
    return actor+' '+msg+(' [托管]' if entry.get('auto') else '')

def load_json(path,default=None):
    try:
        value=json.loads(path.read_text())
        return value if isinstance(value,dict) else ({} if default is None else default)
    except (OSError,ValueError):return {} if default is None else default

def model_lines(decision):
    infer=decision.get('inference') or {}; view=infer.get('observation') or {}
    values=infer.get('display_value_by_seat',infer.get('value_by_seat'))
    if values:
        value='  '.join(f'S{i}={v:+.3f}' if v is not None else f'S{i}=—' for i,v in enumerate(values))
        label='自身预测 + 阵营比例推导' if infer.get('value_training_clock')=='own' else '预测终局分差'
        lines=[f"VALUE  {value}  ({label}；局 {infer.get('game')} / v{infer.get('version')} / step {decision.get('global_step','—')} 决策前)"]
    else:lines=['VALUE  等待首次模型决策；兜底动作不伪造 value']
    if view:
        unseen=' '.join(f'{rank}:{round(row[5]*4)}' for rank,row in zip(RANKS,view['ranks']) if row[5])
        lines+=[f"模型输入  当前牌局 {view['hist_len']} 个出牌/空过事件；合法动作主体 {len([i for i in view['legal_indices'] if i<309])}",
                '未见牌数量（含未公开底牌）  '+unseen]
    return lines

def render_terminal(status,state,decision,width=120,height=28,now=None):
    now=time.time() if now is None else now
    age=max(0,now-status.get('last_http_completed',status.get('time',now)))
    state_current=state.get('id')==status.get('game')
    if not state_current:state={}
    mode=status.get('state','starting'); auto=status.get('our_auto',False)
    health='托管，需要恢复' if auto else '请求延迟/正在恢复' if age>10 else '连接正常'
    lines=[f"DDZ  /  Fortune  ·  QOJ 计分比赛  ·  V5 / step {status.get('global_step','—')}",
        f"{health}  |  {mode}  |  HTTP 心跳 {age:.1f}s  |  本次请求 {status.get('last_request_seconds',0)*1000:.0f}ms  |  PID {status.get('pid','—')}",
        f"比赛 {status.get('match_id','—')}  第 {status.get('match_round','—')}/{status.get('match_rounds',9)} 局  |  对局 {status.get('game','—')} v{status.get('version','—')}  {PHASES.get(status.get('phase'),'等待匹配')}  |  Rating {status.get('rating','—')} 分数 {status.get('score','—')}",
        '─'*max(0,width-1)]
    for i,p in enumerate(state.get('players',())):
        mark='▶' if i==state.get('turn') else ' '
        lines.append(f"{mark} S{i}  {p.get('username','?'):<14} {role(i,state.get('landlord')):<4}  剩 {p.get('count','?'):>2} 张  {'[自己]' if i==state.get('seat') else ''}  {'托管' if p.get('auto') else '手动'}")
    remain=status.get('remaining_ms')
    clock='—' if remain is None else f'{max(0,remain/1000-age):.1f}s'
    lines.extend(['手牌  '+cards(state.get('hand')),
        f"底牌  {cards(state.get('bottom'))}  |  底分 {state.get('bid','—')}  倍数 {state.get('multiplier','—')}  |  当前行动 S{state.get('turn','—')} 剩余 {clock}"])
    if decision:
        latest=decision.get('game')==status.get('game')
        lines.append(f"最近决策  局 {decision.get('game')} v{decision.get('version')} / {decision.get('source')} / {decision.get('decision_seconds',0)*1000:.1f}ms  {action_text(decision.get('endpoint'),decision.get('payload',{}))}"+(' [上一局]' if not latest else ''))
    lines.extend(model_lines(decision))
    history=[e for e in state.get('log',()) if e.get('kind') in ('play','pass','bid','redeal','landlord')]
    lines.append('近期公开事件  '+(' → '.join(event_text(e) for e in history[-3:]) or '等待首个事件'))
    lines.append(f"累计完成 {status.get('games_completed',0)} 局 / {status.get('matches_completed',0)} 场  |  本进程动作 {status.get('accepted_actions',0)} 冲突 {status.get('conflicts',0)} 兜底 {status.get('fallback_decisions',0)} 恢复 {status.get('recoveries',0)}")
    if status.get('last_recovery'):lines.append('最近恢复  '+str(status['last_recovery']))
    lines.append('下方窗格: 实时调试日志  ·  dashboard.html: 完整模型输入/历史/value  ·  Ctrl-B D 脱离')
    return '\n'.join(clip(line,max(0,width-1)) for line in lines[:max(1,height-1)])

def render_html(status,state,decision):
    if state.get('id')!=status.get('game'):state={}
    esc=lambda value:html.escape(str(value),quote=True)
    infer=decision.get('inference') or {}; view=infer.get('observation') or {}
    age=max(0,time.time()-status.get('last_http_completed',status.get('time',time.time())))
    card_html=lambda ids:''.join('<span class="card">'+esc(rank)+'</span>' for rank in cards(ids).split())
    rows=''.join('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in (
        '▶' if i==state.get('turn') else '',f'S{i}',p.get('username','?'),
        role(i,state.get('landlord')),p.get('count','?'),'自己' if i==state.get('seat') else '',
        '托管' if p.get('auto') else '手动'))+'</tr>' for i,p in enumerate(state.get('players',())))
    values=infer.get('display_value_by_seat',infer.get('value_by_seat',[]))
    value_html=''.join('<div class="metric"><small>'+esc(f'S{i} '+('自身 value' if i==infer.get('seat') else '阵营推导' if infer.get('value_training_clock')=='own' else '预期分差'))+'</small><strong>'+(f'{v:+.3f}' if v is not None else '—')+'</strong></div>' for i,v in enumerate(values)) or '<p>等待模型决策</p>'
    rank_rows=''.join('<tr><th>'+esc(rank)+'</th>'+''.join(f'<td>{round(x*4,3):g}</td>' for x in row[:6])+'</tr>' for rank,row in zip(RANKS,view.get('ranks',())))
    history=''.join('<li>'+esc(event_text(e))+'</li>' for e in state.get('log',()) if e.get('kind') in ('bid','landlord','redeal','play','pass','auto_on','auto_off'))
    raw_view=esc(json.dumps(view,ensure_ascii=False,indent=2))
    return f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="3"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fortune · DDZ live</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#0b1423;color:#e2e8f0;font:15px system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:center}}h1{{font-size:28px}}h2{{font-size:18px}}small,.muted{{color:#93a4bc}}.badge{{background:#153c3b;color:#5eead4;padding:8px 14px;border-radius:24px}}.grid{{display:grid;grid-template-columns:1.5fr 1fr;gap:18px}}section{{background:#132034;border:1px solid #26374f;border-radius:16px;padding:20px;margin:18px 0}}.card{{display:inline-block;background:#f1f5f9;color:#182336;font-weight:750;padding:10px 9px;margin:3px;border-radius:6px;min-width:34px;text-align:center}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #26374f}}.metrics{{display:flex;gap:20px;flex-wrap:wrap}}.metric{{flex:1}}.metric strong{{display:block;color:#67e8f9;font-size:30px}}pre{{overflow:auto;max-height:480px;background:#0b1423;padding:12px;border-radius:8px;font-size:12px}}ol{{max-height:480px;overflow:auto;line-height:1.9}}.action{{font-size:20px;color:#fcd34d}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}</style>
<main><header><div><small>QOJ / 连续计分比赛 / CPU V5</small><h1>Fortune · 实时牌桌</h1></div><span class="badge">{esc(status.get('state'))} · 心跳 {age:.1f}s</span></header>
<p class="muted">比赛 {esc(status.get('match_id'))} · 第 {esc(status.get('match_round'))}/{esc(status.get('match_rounds',9))} 局 · 对局 {esc(status.get('game'))} v{esc(status.get('version'))} · Rating {esc(status.get('rating'))} · 权重 step {esc(status.get('global_step'))}</p>
<div class="grid"><div><section><h2>当前牌局 · {esc(PHASES.get(state.get('phase'),'等待'))}</h2><table>{rows}</table><h2>自己的手牌</h2>{card_html(state.get('hand'))}<p>公开底牌 {card_html(state.get('bottom'))}</p><p>底分 {esc(state.get('bid'))} · 倍数 {esc(state.get('multiplier'))} · 行动座位 S{esc(state.get('turn'))}</p></section>
<section><h2>模型最近一次决策</h2><p class="action">{esc(action_text(decision.get('endpoint'),decision.get('payload',{})))}</p><p class="muted">局 {esc(decision.get('game'))} / v{esc(decision.get('version'))} · 来源 {esc(decision.get('source'))} · 决策耗时 {decision.get('decision_seconds',0)*1000:.1f}ms</p><div class="metrics">{value_html}</div><p class="muted">Value 是预测终局原始分差，不是胜率。当前同座位 GAE 主要监督自身 value；其他座位按地主/农民的结算比例推导，叫分时仅显示自身。模型看不到对手手牌。数值来自注明版本的决策前局面，等待别人行动时不会重新推理。</p></section>
<section><h2>模型实际输入 · 局 {esc(infer.get('game'))} / v{esc(infer.get('version'))}</h2><p>当前牌局出牌/空过时序 {esc(view.get('hist_len','—'))} 个事件；旧拍卖信息编码于 context 向量。下家/门板和历史座位均按自身位置重排。</p><table><tr><th>牌面</th><th>手牌</th><th>底牌</th><th>自己已出</th><th>下位已出</th><th>上位已出</th><th>未见牌</th></tr>{rank_rows}</table><details><summary>查看完整张量：ranks / context / 当前牌局变长 history / legal_indices</summary><pre>{raw_view}</pre></details><p class="muted">legal_indices 是动作主体/叫分的有效掩码索引，不等于完整组合动作数量；未见牌含未公开底牌。模型不接收对手隐藏手牌。</p></section></div>
<div><section><h2>运行健康</h2><p>已完成 {esc(status.get('games_completed',0))} 局 / {esc(status.get('matches_completed',0))} 场</p><p>本进程成功动作 {esc(status.get('accepted_actions',0))} · 冲突 {esc(status.get('conflicts',0))} · 兜底 {esc(status.get('fallback_decisions',0))} · 恢复 {esc(status.get('recoveries',0))}</p><p>请求 {status.get('last_request_seconds',0)*1000:.0f}ms · HTTP 心跳 {age:.1f}s · 托管 {esc(status.get('our_auto'))}</p><p>最近恢复：{esc(status.get('last_recovery','无'))}</p><p class="muted">每 3 秒刷新；完整请求、动作、异常日志见 events.jsonl / console.log。</p></section><section><h2>完整公开事件</h2><ol>{history}</ol></section></div></div></main></html>'''
