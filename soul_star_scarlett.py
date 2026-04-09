#!/usr/bin/env python3
"""郝思嘉 · 灵魂星图 — character spec wrapper using soul_star API."""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/soul-star'))

from pathlib import Path
from soul_star import generate_soul_star
from soul_star._engine import REL_THRESH

OUT = Path.home() / 'Desktop' / 'SoulStarMaps' / '斯佳丽'

# ── Domain intensity curves (pct 0→100) ───────────────────────────
D0_PASSION    = [(0,2.5),(8,4.0),(18,7.0),(30,9.0),(45,9.5),(65,9.5),(80,9.5),(100,9.0)]
D1_SURVIVAL   = [(0,1.5),(14,5.5),(22,8.0),(35,8.5),(50,9.0),(75,9.0),(100,9.0)]
D2_CHARM      = [(0,2.5),(12,4.5),(28,7.0),(50,7.5),(65,7.5),(80,7.0),(100,6.0)]
D3_COLD_CALC  = [(0,1.5),(10,3.5),(22,6.5),(38,8.0),(55,8.5),(70,8.5),(85,7.5),(100,8.0)]
D4_LONELINESS = [(0,1.0),(15,2.5),(28,5.5),(45,7.5),(60,8.0),(75,8.5),(90,9.0),(100,9.0)]

# ── Relation influence curves ────────────────────────────────────
R0_TARA     = [(0,3.0),(18,6.5),(28,8.5),(38,9.5),(50,9.5),(65,9.5),(100,9.5)]
R1_RHETT    = [(0,0.5),(8,2.0),(20,4.5),(32,7.0),(44,8.5),(55,9.0),(65,9.5),(78,8.5),(88,7.5),(100,6.5)]
R2_ASHLEY   = [(0,2.0),(7,3.5),(14,6.0),(22,8.0),(32,9.0),(50,9.0),(62,8.5),(72,6.5),(80,4.5),(88,3.0),(95,2.2),(100,2.0)]
R3_TOMORROW = [(0,1.5),(20,3.5),(38,5.5),(55,7.0),(68,7.8),(80,8.5),(90,9.0),(100,9.0)]
R4_SOUTH    = [(0,2.5),(18,5.5),(30,7.5),(48,7.5),(62,6.5),(75,5.5),(88,5.0),(100,5.0)]

# ── Pattern depth curves ─────────────────────────────────────────
P0_SCORPIUS   = [(0,1.5),(18,5.0),(32,8.0),(48,9.0),(65,9.5),(80,9.5),(100,9.5)]
P1_LEO        = [(0,1.5),(15,4.0),(30,7.5),(50,8.0),(68,7.0),(80,6.5),(100,5.5)]
P2_CASSIOPEIA = [(0,1.0),(20,3.5),(38,7.0),(55,8.0),(70,8.5),(85,8.5),(100,8.5)]
P3_CAPRICORN  = [(0,1.0),(25,3.5),(42,7.0),(60,8.0),(75,7.5),(90,8.0),(100,8.0)]

# ── Ecology element curves ───────────────────────────────────────
E_MELANIE = [(0,0),(12,0),(18,5.5),(30,7.8),(55,8.5),(70,7.5),(78,0)]
E_BONNIE  = [(0,0),(55,0),(60,7.0),(65,9.2),(72,8.5),(76,0)]
E_CHARLES = [(0,0),(8,0),(11,5.0),(14,5.5),(18,0)]
E_FRANK   = [(0,0),(30,0),(34,5.5),(40,6.8),(46,5.5),(50,0)]
E_SELF    = [(0,0),(85,0),(90,5.0),(96,7.8),(100,8.5)]
E_YOUNG   = [(0,4.5),(8,8.2),(16,7.5),(25,0)]
E_AWAKEN  = [(78,0),(83,5.2),(88,7.8),(100,7.8)]
E_WARFIRE = [(0,0),(16,0),(22,6.5),(28,8.8),(36,9.2),(42,7.5),(45,0)]
E_LONELY  = [(0,0),(82,0),(86,5.0),(90,7.2),(96,8.2),(100,8.8)]

# ── Atmosphere curves ────────────────────────────────────────────
BG_TEMP_CURVE = [
    (0,0.72),(18,0.52),(25,0.11),(38,0.22),
    (50,0.62),(65,0.80),(75,0.42),(80,0.07),
    (88,0.32),(100,0.60),
]
AURORA_CURVE = [
    (0,0.0),(20,0.0),(25,0.55),(35,0.85),(45,0.30),
    (55,0.0),(76,0.0),(80,0.90),(88,0.40),(100,0.18),
]

# ── Story beats ──────────────────────────────────────────────────
STORY_BEATS = [
    (0,  '序幕', '佐治亚州的夏日·一切将开始'),
    (7,  '初见', '烽火前夕·艾希礼的宣告'),
    (14, '战鼓', '南北战争爆发·世界颠覆'),
    (20, '撤退', '亚特兰大沦陷·烈火中逃离'),
    (26, '废墟', '回到塔拉·发誓不再挨饿'),
    (32, '归来', '重建时代·艾希礼的身影'),
    (38, '谋算', '锯木厂·以艾希礼之名'),
    (44, '婚姻', '瑞德求婚·复杂的结合'),
    (50, '顶峰', '郝思嘉的帝国·表面光鲜'),
    (56, '裂痕', '波妮蓝·命运的转折'),
    (62, '错觉', '艾希礼的怀抱·被人误解'),
    (68, '觉醒', '幻象开始动摇·不过如此'),
    (74, '崩溃', '梅兰妮之死·艾希礼的真相'),
    (80, '幻灭', '我从未爱过艾希礼'),
    (86, '绝望', '瑞德的最后一句话'),
    (92, '独行', '明天再想吧·一个人的路'),
    (100,'真我', '我要回塔拉·大地永存'),
]

# ── Full spec ────────────────────────────────────────────────────
SPEC = {
    'name': '郝思嘉', 'title': 'Scarlett_OHara',
    'gif_name': 'Scarlett_生态星图.gif',
    'W': 26.0, 'H': 26.0, 'n_frames': 60, 'gif_duration_ms': 300,
    'bg_temp_curve': BG_TEMP_CURVE,
    'aurora_curve':  AURORA_CURVE,
    'domains': [
        {'cx':12.35,'cy':11.70,'rx':8.06,'ry':5.33,'ang': 15,
         'dk':'#1a0003','mk':'#8c0022','bk':'#f03055','seed':11,'name':'热烈渴望'},
        {'cx':18.85,'cy': 8.45,'rx':5.46,'ry':3.64,'ang':-35,
         'dk':'#190800','mk':'#8a4010','bk':'#f07028','seed':22,'name':'生存本能'},
        {'cx': 6.24,'cy': 5.20,'rx':3.90,'ry':2.34,'ang': 70,
         'dk':'#160012','mk':'#6a0070','bk':'#e058e0','seed':55,'name':'蛊惑魅力'},
        {'cx': 5.20,'cy':18.85,'rx':4.94,'ry':3.12,'ang': 55,
         'dk':'#001418','mk':'#005058','bk':'#00c8a0','seed':33,'name':'冷静算计'},
        {'cx':22.75,'cy':17.55,'rx':4.42,'ry':2.60,'ang':-20,
         'dk':'#04000e','mk':'#180068','bk':'#4018d0','seed':44,'name':'深层孤独'},
    ],
    'relations': [
        {'name':'塔拉',  'domain':0,'angle':215,'color':'#ff4520'},
        {'name':'瑞德',  'domain':2,'angle': 20,'color':'#ffb060'},
        {'name':'艾希礼','domain':3,'angle':310,'color':'#a8c4ff'},
        {'name':'明日',  'domain':4,'angle':195,'color':'#cce0ff'},
        {'name':'南方',  'domain':1,'angle':135,'color':'#fff4d0'},
    ],
    'patterns': [
        {'tmpl':'scorpius','name_cn':'天蝎座','cx':13.00,'cy':21.45,'sx':6.50,'sy':7.15,
         'soul':{'Antares':('热烈渴望','#ff6040'),'Epsilon':('生存意志','#ffb060'),'Shaula':('占有欲望','#c0d4ff')}},
        {'tmpl':'leo','name_cn':'狮子座','cx':8.45,'cy':14.95,'sx':5.46,'sy':4.94,
         'soul':{'Regulus':('塔拉骄傲','#d8ecff'),'Denebola':('冷静算计','#a0c8ff')}},
        {'tmpl':'cassiopeia','name_cn':'仙后座','cx':20.15,'cy':13.65,'sx':4.55,'sy':2.60,
         'soul':{'Schedar':('不灭之志','#ffb060'),'Gamma':('深层孤独','#c0d8ff')}},
        {'tmpl':'capricorn','name_cn':'摩羯座','cx':17.55,'cy': 5.85,'sx':5.85,'sy':4.55,
         'soul':{'Algedi':('操纵心计','#f0e0a0'),'Deneb_Alg':('冷酷现实','#a0b8ff')}},
    ],
    'domain_curves':   [D0_PASSION, D1_SURVIVAL, D2_CHARM, D3_COLD_CALC, D4_LONELINESS],
    'relation_curves': [R0_TARA, R1_RHETT, R2_ASHLEY, R3_TOMORROW, R4_SOUTH],
    'pattern_curves':  [P0_SCORPIUS, P1_LEO, P2_CASSIOPEIA, P3_CAPRICORN],
    'story_beats': STORY_BEATS,
    'lifecycle_label': {
        'name': '艾希礼',
        'curve': R2_ASHLEY,
        'thresholds': (REL_THRESH, 4.5, 1.5),
        'tier_names': ('恒星', '原始星云', '碎片', '消散'),
        'color': '#a8c4ff',
    },
    'ecology': [
        {'name':'梅兰妮','etype':'star','birth':12,'death':78,
         'curve':E_MELANIE,'color':'#c8e890','excl_r':2.5},
        {'name':'邦妮',  'etype':'star','birth':55,'death':76,
         'curve':E_BONNIE, 'color':'#f0b0c0','excl_r':2.5},
        {'name':'查尔斯','etype':'star','birth':8, 'death':18,
         'curve':E_CHARLES,'color':'#b8b8d0','excl_r':2.5},
        {'name':'弗兰克','etype':'star','birth':30,'death':50,
         'curve':E_FRANK,  'color':'#c8a070','excl_r':2.5},
        {'name':'自我',  'etype':'star','birth':85,'death':100,
         'curve':E_SELF,   'color':'#f0f0e0','excl_r':2.5},
        {'name':'青春娱乐','etype':'const','birth':0,'death':25,
         'curve':E_YOUNG,'name_cn':'欢愉','tmpl':'young_fun',
         'sx':3.2,'sy':2.5,'soul':{'B':('时尚','#ffd080'),'D':('宴会','#ffc060')},
         'excl_r':3.8},
        {'name':'醒悟真我','etype':'const','birth':78,'death':100,
         'curve':E_AWAKEN,'name_cn':'醒悟','tmpl':'crown',
         'sx':3.0,'sy':2.5,'soul':{'E':('自知','#f0e8d0'),'B':('明日','#f8f4e8')},
         'excl_r':3.8},
        {'name':'战火创伤','etype':'neb','birth':16,'death':45,
         'curve':E_WARFIRE,'rx':4.0,'ry':2.6,'ang':28,
         'dk':'#180400','mk':'#880018','bk':'#ff4820','seed':77,'excl_r':4.5},
        {'name':'晚秋孤寂','etype':'neb','birth':82,'death':100,
         'curve':E_LONELY,'rx':3.2,'ry':2.0,'ang':-25,
         'dk':'#030810','mk':'#152048','bk':'#2848a0','seed':88,'excl_r':3.8},
    ],
}


def main():
    generate_soul_star(SPEC, OUT)


if __name__ == '__main__':
    main()
