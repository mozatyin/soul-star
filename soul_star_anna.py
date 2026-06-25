#!/usr/bin/env python3
"""安娜·卡列尼娜 · 灵魂星图 — character spec wrapper using soul_star API."""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/soul-star'))

from pathlib import Path
from soul_star import generate_soul_star
from soul_star._engine import REL_THRESH

OUT = Path.home() / 'Desktop' / 'SoulStarMaps' / '安娜卡列尼娜'

# ── Domain intensity curves (pct 0→100) ───────────────────────────
# 托尔斯泰《安娜·卡列尼娜》8部全程映射
D0_TRAPPED   = [(0,5.0),(20,4.0),(35,3.5),(55,5.5),(65,7.0),(75,8.0),(85,9.0),(100,9.5)]
D1_PASSION   = [(0,1.0),(15,4.5),(25,7.5),(38,9.5),(50,9.0),(62,8.5),(72,7.0),(82,5.0),(92,3.0),(100,1.0)]
D2_MATERNAL  = [(0,3.5),(25,4.5),(42,5.5),(52,7.0),(62,8.5),(72,9.0),(82,8.5),(92,9.0),(100,8.5)]
D3_SOCIETY   = [(0,6.5),(18,9.0),(35,8.5),(50,7.0),(60,5.0),(70,2.5),(80,1.5),(90,1.0),(100,0.8)]
D4_JEALOUSY  = [(0,1.0),(35,2.0),(55,3.5),(68,5.5),(78,7.5),(85,9.0),(92,9.5),(97,9.8),(100,9.9)]

# ── Relation influence curves ────────────────────────────────────
# 弗龙斯基 — THE lifecycle star: 上升→顶峰→消散
R0_VRONSKY   = [(0,0.5),(15,3.5),(25,7.0),(35,9.0),(45,9.5),(55,9.0),(65,8.5),(75,7.5),(82,6.0),(90,4.0),(96,2.5),(100,0.5)]
R1_KARENIN   = [(0,7.5),(20,6.5),(35,5.0),(50,8.0),(58,6.5),(68,4.0),(78,2.5),(90,2.0),(100,2.0)]
R2_SERYOZHA  = [(0,5.5),(30,6.0),(50,7.0),(62,8.5),(72,9.0),(82,8.5),(90,9.0),(100,9.0)]
R3_DOLLY     = [(0,5.0),(20,6.0),(35,6.5),(55,6.0),(70,5.5),(82,5.0),(100,4.5)]
R4_KITTY     = [(0,4.0),(18,7.0),(28,5.0),(40,3.5),(60,3.0),(80,2.5),(100,2.0)]

# ── Pattern depth curves ─────────────────────────────────────────
P0_SCORPIUS   = [(0,2.0),(20,4.0),(35,7.5),(50,9.0),(68,9.5),(82,9.5),(100,9.5)]
P1_LEO        = [(0,7.0),(25,8.5),(45,7.0),(60,5.5),(72,3.5),(82,2.0),(100,1.5)]
P2_CASSIOPEIA = [(0,6.5),(20,9.0),(38,8.0),(55,6.0),(68,3.0),(78,1.5),(100,1.0)]
P3_CAPRICORN  = [(0,2.0),(40,3.5),(58,5.5),(72,7.0),(82,8.5),(90,9.0),(100,9.5)]

# ── Ecology element curves ───────────────────────────────────────
E_LEVIN      = [(0,3.0),(20,3.5),(40,4.5),(55,5.0),(70,5.0),(85,4.5),(100,5.0)]
E_BALL       = [(12,0),(14,7.0),(16,8.5),(18,8.0),(19,0)]
E_HORSERACE  = [(24,0),(25,6.0),(27,8.5),(29,7.5),(30,0)]
E_ITALY      = [(56,0),(58,6.0),(62,8.0),(66,7.0),(68,0)]
E_SEPARATION = [(45,0),(48,5.0),(55,7.5),(65,8.5),(75,8.0),(85,7.5),(100,8.0)]
E_SOCIETY    = [(0,5.5),(18,8.0),(35,7.5),(50,5.0),(60,3.0),(68,1.5),(70,0)]
E_JEALOUSY   = [(72,0),(76,4.5),(80,7.5),(85,9.0),(90,9.5),(96,9.8),(100,9.9)]

# ── Atmosphere curves ────────────────────────────────────────────
# 俄国冬日宫廷→热恋→冷漠社会→极寒深渊
BG_TEMP_CURVE = [
    (0,0.65),(15,0.72),(28,0.85),(40,0.75),
    (55,0.50),(65,0.30),(75,0.18),(82,0.10),
    (90,0.05),(100,0.02),
]
AURORA_CURVE = [
    (0,0.0),(25,0.0),(35,0.3),(45,0.5),
    (55,0.35),(65,0.15),(72,0.0),
    (78,0.65),(85,0.90),(92,0.80),(100,0.95),
]

# ── Story beats ──────────────────────────────────────────────────
STORY_BEATS = [
    (0,  '婚姻',  '彼得堡·卡列宁家的平静生活'),
    (10, '出行',  '莫斯科·解救哥哥的婚姻'),
    (16, '初遇',  '舞会·弗龙斯基的出现·吉提的失落'),
    (24, '磁场',  '火车站·命运的交汇'),
    (35, '燃烧',  '彼得堡·激情无法抑制'),
    (44, '出轨',  '彻底背叛婚姻'),
    (50, '坦白',  '赛马场坠马·产后大病·向卡列宁坦白'),
    (58, '宽恕',  '丈夫的宽恕·流亡意大利'),
    (66, '孤立',  '彼得堡社交圈的冷漠封锁'),
    (74, '裂变',  '弗龙斯基的疏远·妒火爆发'),
    (82, '深渊',  '精神崩溃·猜疑蔓延'),
    (90, '幻灭',  '一切都是谎言和欺骗'),
    (100,'车轮',  '雅乌扎车站·铁轨之下'),
]

# ── Full spec ────────────────────────────────────────────────────
SPEC = {
    'name': '安娜·卡列尼娜', 'title': 'Anna_Karenina',
    'gif_name': 'Anna_Karenina_灵魂星图.gif',
    'W': 26.0, 'H': 26.0, 'n_frames': 60, 'gif_duration_ms': 300,
    'bg_temp_curve': BG_TEMP_CURVE,
    'aurora_curve':  AURORA_CURVE,
    'domains': [
        # 0 被囚的灵魂 — left-center, cold grey-blue
        {'cx': 6.5, 'cy':13.5,'rx':4.5,'ry':2.8,'ang': 30,
         'dk':'#060810','mk':'#141a40','bk':'#2838a0','seed':301,'name':'被囚的灵魂'},
        # 1 炽烈之爱 — center, deep crimson
        {'cx':13.5, 'cy':12.5,'rx':8.0,'ry':5.2,'ang': 10,
         'dk':'#140002','mk':'#8a0015','bk':'#e02040','seed':302,'name':'炽烈之爱'},
        # 2 母性牵绊 — upper right, warm amber
        {'cx':20.5, 'cy':19.5,'rx':4.2,'ry':2.6,'ang':-20,
         'dk':'#140800','mk':'#804010','bk':'#e08030','seed':303,'name':'母性牵绊'},
        # 3 社会荣耀 — upper center, cold gold
        {'cx':13.5, 'cy':21.5,'rx':5.0,'ry':3.0,'ang': 15,
         'dk':'#0e0a00','mk':'#706000','bk':'#d4b020','seed':304,'name':'社会荣耀'},
        # 4 深渊妒火 — right, dark purple
        {'cx':21.5, 'cy': 8.5,'rx':4.0,'ry':2.5,'ang':-35,
         'dk':'#080010','mk':'#300050','bk':'#7020b0','seed':305,'name':'深渊妒火'},
    ],
    'relations': [
        {'name':'弗龙斯基','domain':1,'angle': 25,'color':'#ff6080'},
        {'name':'卡列宁',  'domain':0,'angle':200,'color':'#8090b8'},
        {'name':'谢廖沙',  'domain':2,'angle':150,'color':'#ffd090'},
        {'name':'多丽',    'domain':3,'angle':280,'color':'#c8b840'},
        {'name':'吉提',    'domain':3,'angle': 80,'color':'#d0e8c0'},
    ],
    'patterns': [
        {'tmpl':'scorpius','name_cn':'天蝎座','cx': 8.0,'cy': 5.5,'sx':5.5,'sy':6.0,
         'soul':{'Antares':('激情之焰','#ff4040'),'Shaula':('占有欲望','#c040ff')}},
        {'tmpl':'leo','name_cn':'狮子座','cx':20.5,'cy': 5.5,'sx':5.0,'sy':4.5,
         'soul':{'Regulus':('贵族尊严','#d8ecff'),'Denebola':('独立意志','#a0c8ff')}},
        {'tmpl':'cassiopeia','name_cn':'仙后座','cx': 6.5,'cy':21.5,'sx':4.5,'sy':2.5,
         'soul':{'Schedar':('上流社会','#f8e080'),'Gamma':('虚荣之冠','#e8d860')}},
        {'tmpl':'capricorn','name_cn':'摩羯座','cx':21.5,'cy':21.5,'sx':5.5,'sy':4.0,
         'soul':{'Algedi':('绝望深渊','#9060e0'),'Deneb_Alg':('冷酷命运','#6040c0')}},
    ],
    'domain_curves':   [D0_TRAPPED, D1_PASSION, D2_MATERNAL, D3_SOCIETY, D4_JEALOUSY],
    'relation_curves': [R0_VRONSKY, R1_KARENIN, R2_SERYOZHA, R3_DOLLY, R4_KITTY],
    'pattern_curves':  [P0_SCORPIUS, P1_LEO, P2_CASSIOPEIA, P3_CAPRICORN],
    'story_beats': STORY_BEATS,
    'lifecycle_label': {
        'name': '弗龙斯基',
        'curve': R0_VRONSKY,
        'thresholds': (REL_THRESH, 4.5, 1.5),
        'tier_names': ('恒星', '原始星云', '碎片', '消散'),
        'color': '#ff6080',
    },
    'ecology': [
        # 列文 — 精神对照，全程若隐若现
        {'name':'列文',   'etype':'star','birth':0,  'death':100,
         'curve':E_LEVIN,  'color':'#90c890','excl_r':2.5},
        # 莫斯科舞会 — 命运转折点
        {'name':'莫斯科舞会','etype':'const','birth':12,'death':20,
         'curve':E_BALL,'name_cn':'舞会','tmpl':'young_fun',
         'sx':2.8,'sy':2.2,'soul':{'B':('吉提失落','#d0e8c0'),'D':('弗龙斯基目光','#ff8090')},
         'excl_r':3.2},
        # 赛马场事故 — 公开危机
        {'name':'赛马场',  'etype':'star','birth':24,'death':30,
         'curve':E_HORSERACE,'color':'#e8c040','excl_r':2.5},
        # 意大利流亡 — 短暂幸福
        {'name':'意大利',  'etype':'star','birth':56,'death':68,
         'curve':E_ITALY,   'color':'#60c8b0','excl_r':2.5},
        # 母子分离之痛 — 贯穿后半段
        {'name':'谢廖沙分离','etype':'star','birth':45,'death':100,
         'curve':E_SEPARATION,'color':'#ffd090','excl_r':2.5},
        # 莫斯科社交圈 — 前半段的氛围底色
        {'name':'社交圈',  'etype':'neb','birth':0, 'death':70,
         'curve':E_SOCIETY,
         'rx':4.2,'ry':2.8,'ang':20,
         'dk':'#0a0a04','mk':'#505010','bk':'#a09020','seed':311,'excl_r':4.0},
        # 嫉妒黑洞 — 后期爆发的深渊
        {'name':'嫉妒深渊', 'etype':'neb','birth':72,'death':100,
         'curve':E_JEALOUSY,
         'rx':3.5,'ry':2.2,'ang':-30,
         'dk':'#080010','mk':'#300050','bk':'#7020b0','seed':312,'excl_r':4.2},
    ],
}


def main():
    generate_soul_star(SPEC, OUT)


if __name__ == '__main__':
    main()
