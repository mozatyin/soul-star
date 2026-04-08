#!/usr/bin/env python3
"""
Soul Star Map  v28 — Full Lifecycle Animation + Ecology + Sky Atmosphere
郝思嘉完整心路历程 · 从第一句话到最后一刻
─────────────────────────────────────────────
每帧 = 约100句故事话语 · 共60帧 · 生成GIF动画
位置完全锁定：所有星体只有亮暗变化，绝不漂移
艾希礼始终在左上角，从诞生到消散
v28新增：生态元素系统 + 天空大气层
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPoly
from scipy.ndimage import gaussian_filter
from PIL import Image
import time
from pathlib import Path

plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS',
    'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

W, H = 26.0, 26.0
OUT = Path.home() / 'Desktop' / 'SoulStarMaps' / '斯佳丽'

# Animation params: smaller canvas for speed, reduce fractal complexity
FIG_SZ = 9.1   # scaled with W: 7.0 * (26/20) — keeps star visual size constant
DPI    = 100   # 910×910 px per frame

P = {
    'bg_n':2000, 'bg_maxr':0.260, 'bg_alpha':0.290, 'bg_band':1.327,
    'neb_gamma':2.642, 'neb_amax':0.76,  'neb_blur_k':0.14,
    'neb_amult':9.5,   'neb_n_oct':3.0,  'neb_env_n':0.310,   # 3 octaves (faster)
    'cl_alpha':0.430,
    'sf_outer_r':0.340, 'sf_alpha':0.960, 'sf_lsz':19.0,
    'sf_glow':7.5, 'sf_ray':9.5,
    'cn_soul_r':0.118, 'cn_soul_a':0.870, 'cn_reg_a':0.620,
}

TMPL = {
    'scorpius':{'stars':{'Graffias':((0.08,0.90),2,'#a8c4ff'),'Dschubba':((0.22,0.95),3,'#a0bcff'),'Sigma':((0.35,0.82),3,'#aec8ff'),'Antares':((0.42,0.70),1,'#ff4520'),'Tau':((0.50,0.60),3,'#a0bcff'),'Epsilon':((0.58,0.48),2,'#ffb060'),'Mu':((0.65,0.37),3,'#b0c8ff'),'Theta':((0.78,0.28),3,'#f8e8c0'),'Shaula':((0.92,0.48),1,'#b0ccff'),'Lesath':((0.88,0.54),2,'#b8d0ff')},'lines':[('Graffias','Dschubba'),('Dschubba','Sigma'),('Sigma','Antares'),('Antares','Tau'),('Tau','Epsilon'),('Epsilon','Mu'),('Mu','Theta'),('Theta','Shaula'),('Shaula','Lesath')]},
    'leo':{'stars':{'Regulus':((0.18,0.12),1,'#cce0ff'),'Eta_Leo':((0.20,0.38),3,'#f0f4ff'),'Gamma_Leo':((0.28,0.60),2,'#ffb060'),'Zeta_Leo':((0.40,0.78),3,'#fff0c0'),'Mu_Leo':((0.55,0.76),3,'#ffb860'),'Denebola':((0.88,0.32),2,'#d0e8ff')},'lines':[('Regulus','Eta_Leo'),('Eta_Leo','Gamma_Leo'),('Gamma_Leo','Zeta_Leo'),('Zeta_Leo','Mu_Leo'),('Mu_Leo','Denebola')]},
    'cassiopeia':{'stars':{'Caph':((0.02,0.52),2,'#f8f0a0'),'Schedar':((0.25,0.98),2,'#ffb060'),'Gamma':((0.50,0.42),2,'#b0c8ff'),'Ruchbah':((0.75,0.92),2,'#e8e8ff'),'Segin':((0.98,0.38),3,'#b0c8ff')},'lines':[('Caph','Schedar'),('Schedar','Gamma'),('Gamma','Ruchbah'),('Ruchbah','Segin')]},
    'capricorn':{'stars':{'Algedi':((0.05,0.88),3,'#fff0d0'),'Dabih':((0.20,0.80),2,'#fff0c0'),'Omega_Cap':((0.50,0.62),3,'#f8f8f0'),'Zeta_Cap':((0.82,0.58),3,'#f0f0e8'),'Nashira':((0.72,0.42),3,'#f0e8d0'),'Deneb_Alg':((0.90,0.22),2,'#a8c4ff')},'lines':[('Algedi','Dabih'),('Dabih','Omega_Cap'),('Omega_Cap','Zeta_Cap'),('Zeta_Cap','Nashira'),('Nashira','Deneb_Alg')]},
}

TMPL['young_fun'] = {
    'stars': {
        'A': ((0.12, 0.50), 3, '#ffd080'),
        'B': ((0.38, 0.92), 2, '#ffb840'),
        'C': ((0.68, 0.82), 3, '#ffc860'),
        'D': ((0.90, 0.42), 2, '#ffe090'),
        'E': ((0.50, 0.08), 3, '#ffd060'),
    },
    'lines': [('A','B'),('B','C'),('C','D'),('A','E'),('E','D')]
}
TMPL['crown'] = {
    'stars': {
        'A': ((0.08, 0.62), 2, '#f0e8c0'),
        'B': ((0.32, 0.96), 2, '#f8f0d0'),
        'C': ((0.68, 0.96), 2, '#f0e8c0'),
        'D': ((0.92, 0.62), 2, '#f8f0d0'),
        'E': ((0.50, 0.22), 1, '#fff8e0'),
    },
    'lines': [('A','B'),('B','E'),('E','C'),('C','D'),('A','D')]
}

DOMAIN_THRESH = 6.0
REL_THRESH    = 7.5
PAT_THRESH    = 7.0

def _neb_sc(v):  return (v / 10.0) ** 0.60
def _neb_am(v):  return (v / 10.0) ** 0.80 * 0.95
def _rel_mg(v):  return round(min(9.5, max(7.0, 7.5 + (v - 7.5) * 0.87)), 1)

# ══════════════════════════════════════════════════════════════════
# VISUAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def hex2rgb(h): return tuple(int(h[i:i+2],16)/255.0 for i in (1,3,5))
def _pts(fs): return (fs/W)*72.0
def _s(r,fs): return max(0.1,(r*_pts(fs)*2.0)**2)

def draw_glow(ax,x,y,r,col,alpha,fs,z):
    for frac,al in [(1.00,0.06),(0.55,0.18),(0.25,0.38)]:
        ax.scatter(x,y,s=_s(r*frac,fs),c=[col],alpha=float(al*alpha),linewidths=0,zorder=z)

def draw_four_point(ax,x,y,outer_r,col,alpha,z):
    ir=outer_r*0.27; rgb=hex2rgb(col); v=[]
    for i in range(4):
        ao=np.pi/2-i*np.pi/2; ai=ao-np.pi/4
        v.append((x+np.cos(ao)*outer_r,y+np.sin(ao)*outer_r))
        v.append((x+np.cos(ai)*ir,y+np.sin(ai)*ir))
    ax.add_patch(MplPoly(v,closed=True,facecolor=(*rgb,float(alpha)),linewidth=0,zorder=z))

def draw_six_ray(ax,x,y,rlen,col,alpha,z):
    for i in range(6):
        ang=i*np.pi/3-np.pi/6; lw=1.0 if i%2==0 else 0.45
        for t0,t1,am in [(0.0,0.22,0.55),(0.22,0.55,0.18),(0.55,1.0,0.05)]:
            ax.plot([x+np.cos(ang)*rlen*t0,x+np.cos(ang)*rlen*t1],
                    [y+np.sin(ang)*rlen*t0,y+np.sin(ang)*rlen*t1],
                    '-',color=col,alpha=float(am*alpha),lw=lw,
                    solid_capstyle='round',zorder=z)

def draw_star_full(ax,x,y,outer_r,col,alpha,fs,glow_mult=7.5,ray_mult=9.5,z=7):
    draw_glow(ax,x,y,outer_r*glow_mult,col,alpha,fs,z-0.2)
    draw_six_ray(ax,x,y,outer_r*ray_mult,col,alpha,z)
    draw_glow(ax,x,y,outer_r*2.0,col,alpha*0.88,fs,z)
    draw_four_point(ax,x,y,outer_r,col,alpha*0.93,z+0.1)
    ax.scatter(x,y,s=_s(outer_r*0.22,fs),c=['#FFFDF5'],alpha=0.98,linewidths=0,zorder=z+0.2)

def draw_star_small(ax,x,y,outer_r,col,alpha,fs,z=5):
    draw_glow(ax,x,y,outer_r*3.0,col,alpha*0.50,fs,z-0.1)
    draw_six_ray(ax,x,y,outer_r*4.5,col,alpha*0.65,z)
    draw_four_point(ax,x,y,outer_r,col,alpha*0.90,z+0.1)
    ax.scatter(x,y,s=_s(outer_r*0.25,fs),c=['#FFFDF5'],alpha=0.95,linewidths=0,zorder=z+0.2)

def draw_proto_fragment(ax,x,y,intensity,color,fs,z=2.91):
    frac=max(0,(intensity-1.5)/3.0)
    r=P['sf_outer_r']*0.10*(0.3+0.7*frac)
    draw_glow(ax,x,y,r*4.0,color,frac*0.22,fs,z)
    ax.scatter(x,y,s=_s(r*0.5,fs),c=[color],alpha=float(frac*0.55),linewidths=0,zorder=z+0.1)

def draw_proto_cloud(ax,x,y,rx,ry,intensity,color_hot,fs,z=2.97):
    frac=max(0,(intensity-4.5)/1.5)
    r=max(rx,ry)*_neb_sc(intensity)*0.42
    for f,a in [(1.0,0.04),(0.6,0.10),(0.3,frac*0.20)]:
        ax.scatter(x,y,s=_s(r*f,fs)*3.0,c=[color_hot],alpha=float(a),linewidths=0,zorder=z)

def draw_legacy_stars(ax, pct, fig_sz):
    """Dead ecology elements leave faint memorial stars — once important, now a quiet trace."""
    for el in ECOLOGY_ELEMENTS:
        if el['death'] > pct:
            continue
        pos = ECOLOGY_POSITIONS.get(el['name'])
        if pos is None:
            continue
        ex, ey = pos
        col  = el.get('color', el.get('bk', '#ffffff'))  # star→color, neb→bk
        age  = min(1.0, (pct - el['death']) / 25.0)
        alp  = max(0.12, 0.38 - 0.18*age)
        r    = max(0.030, 0.068 - 0.025*age)
        draw_glow(ax, ex, ey, r*3.5, col, alp*0.42, fig_sz, z=2.70)
        ax.scatter([ex],[ey], s=_s(r,      fig_sz), c=[col],
                  alpha=float(alp),      linewidths=0, zorder=2.72)
        ax.scatter([ex],[ey], s=_s(r*0.42, fig_sz), c=['white'],
                  alpha=float(alp*0.55), linewidths=0, zorder=2.73)

# ── LABEL PLACEMENT ───────────────────────────────────────────────
def find_label_pos(cx,cy,offset,lsz,nchars,fs,placed):
    pts=_pts(fs); lw=nchars*lsz/pts*0.62; lh=lsz/pts*1.25
    dx0=cx-W/2; dy0=cy-H/2
    pref=np.degrees(np.arctan2(dy0,dx0))%360
    cands=[(315,'left','bottom'),(135,'right','top'),(45,'left','top'),(225,'right','bottom'),
           (0,'left','center'),(180,'right','center'),(90,'center','bottom'),(270,'center','top')]
    cands.sort(key=lambda c:min(abs(c[0]-pref),360-abs(c[0]-pref)))
    for ang,ha,va in cands:
        rad=np.radians(ang)
        tx=float(np.clip(cx+np.cos(rad)*offset,0.3,W-0.3))
        ty=float(np.clip(cy+np.sin(rad)*offset,0.3,H-0.3))
        if ha=='right':   bx0=tx-lw;bx1=tx
        elif ha=='center':bx0=tx-lw/2;bx1=tx+lw/2
        else:             bx0=tx;bx1=tx+lw
        if va=='top':     by0=ty-lh;by1=ty
        elif va=='center':by0=ty-lh/2;by1=ty+lh/2
        else:             by0=ty;by1=ty+lh
        if all(bx1+0.08<pb[0] or pb[2]+0.08<bx0 or
               by1+0.08<pb[1] or pb[3]+0.08<by0 for pb in placed):
            placed.append((bx0,by0,bx1,by1)); return tx,ty,ha,va
    placed.append((cx-lw/2,cy-lh/2,cx+lw/2,cy+lh/2))
    return cx,cy-lh,'center','top'

# ── NEBULA RENDERER ───────────────────────────────────────────────
def build_fractal_nebs(p,px,nebs):
    layer=np.zeros((px,px,4),dtype=np.float32)
    rg,cg=np.mgrid[0:px,0:px]
    xs=cg.astype(np.float32); ys=rg.astype(np.float32)
    n_oct=max(2,int(round(p['neb_n_oct']))); gamma=p['neb_gamma']; env_ns=p['neb_env_n']
    for cx_d,cy_d,rx_d,ry_d,ang,dk,mk,bk,amax_b,seed,*_ in nebs:
        cpx,cpy=cx_d/W*px,cy_d/H*px; rx_px,ry_px=rx_d/W*px,ry_d/H*px
        amax=amax_b*p['neb_amax']; rng=np.random.default_rng(seed)
        ca,sa=np.cos(np.radians(ang)),np.sin(np.radians(ang))
        dx,dy=xs-cpx,ys-cpy
        d_ell=np.sqrt(((dx*ca+dy*sa)/rx_px)**2+((-dx*sa+dy*ca)/ry_px)**2)
        radial_c=np.exp(-d_ell**2*1.6)
        rng_e=np.random.default_rng(seed+200)
        esm=gaussian_filter(rng_e.standard_normal((px,px)).astype(np.float32),sigma=max(2.,rx_px*0.65))
        esm=(esm-esm.min())/(esm.max()-esm.min()+1e-8)
        radial=np.clip(radial_c*(1.-env_ns+env_ns*esm),0,1)
        noise=np.zeros((px,px),dtype=np.float32); amp=1.
        for k in range(n_oct):
            nsm=gaussian_filter(rng.standard_normal((px,px)).astype(np.float32),sigma=max(.5,rx_px*.48/(1.85**k)))
            nsm/=(nsm.std()+1e-8); noise+=amp*nsm; amp*=.52
        noise=(noise-noise.min())/(noise.max()-noise.min()+1e-8)
        bright=radial*(noise**gamma)
        for ks in [seed+300,seed+400]:
            rk=np.random.default_rng(ks)
            kx=cpx+rk.uniform(-rx_px*.35,rx_px*.35); ky=cpy+rk.uniform(-ry_px*.35,ry_px*.35)
            bright=np.clip(bright+np.exp(-((xs-kx)**2+(ys-ky)**2)/(2*(max(1.,rx_px*.07))**2))*.50,0,1)
        bmax=bright.max()
        if bmax>1e-8: bright/=bmax
        bright=gaussian_filter(bright,sigma=max(.5,rx_px*p['neb_blur_k']))
        bright/=(bright.max()+1e-8)
        alpha_ch=np.clip(bright*p['neb_amult'],0.,amax)
        def h2a(h): return np.array([int(h[i:i+2],16)/255. for i in (1,3,5)],dtype=np.float32)
        dc,mc,bc=h2a(dk),h2a(mk),h2a(bk)
        t=bright[:,:,np.newaxis]
        col=np.where(t<=.5,dc*(1-t*2)+mc*(t*2),mc*(1-(t-.5)*2)+bc*((t-.5)*2))
        a4=alpha_ch[:,:,np.newaxis]; ea=layer[:,:,3:4]; oa=a4+ea*(1-a4)
        for c in range(3):
            num=col[:,:,c:c+1]*a4+layer[:,:,c:c+1]*ea*(1-a4)
            layer[:,:,c:c+1]=np.where(oa>1e-7,num/np.maximum(oa,1e-7),0)
        layer[:,:,3:4]=oa
    return layer

# ══════════════════════════════════════════════════════════════════
# STORY INTENSITY CURVES  (pct 0→100, linear interpolation)
# Each curve: list of (pct, intensity) breakpoints
# ══════════════════════════════════════════════════════════════════
def lerp(pts, pct):
    """Linear interpolation through (pct, value) breakpoints."""
    pct = max(pts[0][0], min(pts[-1][0], pct))
    for i in range(len(pts)-1):
        p0,v0 = pts[i]; p1,v1 = pts[i+1]
        if p0 <= pct <= p1:
            t = (pct-p0)/(p1-p0) if p1>p0 else 0.0
            return v0 + t*(v1-v0)
    return pts[-1][1]

# ══════════════════════════════════════════════════════════════════
# BACKGROUND ATMOSPHERE  (new in v28)
# ══════════════════════════════════════════════════════════════════
# Sky emotional temperature: 0=arctic cold, 1=warm amber
BG_TEMP_CURVE = [
    (0, 0.72),(18, 0.52),(25, 0.11),(38, 0.22),
    (50, 0.62),(65, 0.80),(75, 0.42),(80, 0.07),
    (88, 0.32),(100, 0.60),
]
# Aurora intensity (meaningful only when temp < 0.45)
AURORA_CURVE = [
    (0, 0.0),(20, 0.0),(25, 0.55),(35, 0.85),(45, 0.30),
    (55, 0.0),(76, 0.0),(80, 0.90),(88, 0.40),(100, 0.18),
]

def render_sky_atmosphere(ax, pct, fig_sz, dpi):
    """Render atmospheric sky gradient + optional aurora. zorder=0.5."""
    temp      = lerp(BG_TEMP_CURVE, pct)
    aurora_s  = lerp(AURORA_CURVE, pct)

    ny, nx = 80, 80
    img = np.zeros((ny, nx, 3), dtype=np.float32)

    # Base gradient top/bottom colors
    if temp <= 0.5:
        t = temp / 0.5
        c_top = np.array([0.020,0.028,0.175])*(1-t) + np.array([0.038,0.020,0.095])*t
        c_bot = np.array([0.028,0.048,0.210])*(1-t) + np.array([0.048,0.028,0.112])*t
    else:
        t = (temp - 0.5) / 0.5
        c_top = np.array([0.038,0.020,0.095])*(1-t) + np.array([0.068,0.028,0.140])*t
        c_bot = np.array([0.048,0.028,0.112])*(1-t) + np.array([0.175,0.075,0.018])*t

    for j in range(ny):
        tf = j / ny   # 0=bottom row, 1=top row (origin='lower')
        img[j] = c_bot*(1-tf) + c_top*tf

    # Warm horizon glow at bottom
    if temp > 0.55:
        g = (temp - 0.55) / 0.45 * 0.20
        for j in range(ny//6):
            fade = 1.0 - j / (ny//6)
            img[j] = np.clip(img[j] + np.array([g*fade*1.8, g*fade*0.45, 0.0]), 0, 1)

    # Aurora bands (cold phases only)
    if aurora_s > 0.03 and temp < 0.45:
        ys_a = np.arange(ny, dtype=float)
        xs_a = np.arange(nx, dtype=float)
        bands = [
            (0.74, np.array([0.16, 0.85, 0.40]), 1.00),  # green
            (0.66, np.array([0.08, 0.38, 0.88]), 0.65),  # blue
            (0.80, np.array([0.60, 0.16, 0.76]), 0.45),  # purple
        ]
        for fy, ac, strength in bands:
            by = fy * ny
            bw = max(2.4, 0.042*ny)
            env  = np.exp(-((ys_a - by)/bw)**2)
            wave = 0.58 + 0.42*np.sin(xs_a*0.22 + pct*0.18)
            contrib = aurora_s * strength * 0.52 * env[:,None] * wave[None,:]
            img += (contrib[:,:,None] * ac[None,None,:]).astype(np.float32)

    img = np.clip(img, 0, 1)
    ax.imshow(img, extent=[0, W, 0, H], origin='lower',
              aspect='auto', zorder=0.5, interpolation='bilinear')

# Domain intensities — what coalesces into nebulae
D0_PASSION   = [(0,2.5),(8,4.0),(18,7.0),(30,9.0),(45,9.5),(65,9.5),(80,9.5),(100,9.0)]
D1_SURVIVAL  = [(0,1.5),(14,5.5),(22,8.0),(35,8.5),(50,9.0),(75,9.0),(100,9.0)]
D2_CHARM     = [(0,2.5),(12,4.5),(28,7.0),(50,7.5),(65,7.5),(80,7.0),(100,6.0)]
D3_COLD_CALC = [(0,1.5),(10,3.5),(22,6.5),(38,8.0),(55,8.5),(70,8.5),(85,7.5),(100,8.0)]
D4_LONELINESS= [(0,1.0),(15,2.5),(28,5.5),(45,7.5),(60,8.0),(75,8.5),(90,9.0),(100,9.0)]

# Relation influences — what crystallizes into surface stars
# 塔拉 — Tara, the land, eternal identity
R0_TARA      = [(0,3.0),(18,6.5),(28,8.5),(38,9.5),(50,9.5),(65,9.5),(100,9.5)]
# 瑞德 — Rhett Butler: slow rise, late peak, then exit
R1_RHETT     = [(0,0.5),(8,2.0),(20,4.5),(32,7.0),(44,8.5),(55,9.0),(65,9.5),(78,8.5),(88,7.5),(100,6.5)]
# 艾希礼 — Ashley: THE lifecycle demo. Rise → peak → slow fade → fragment
R2_ASHLEY    = [(0,2.0),(7,3.5),(14,6.0),(22,8.0),(32,9.0),(50,9.0),(62,8.5),(72,6.5),(80,4.5),(88,3.0),(95,2.2),(100,2.0)]
# 明日 — Tomorrow (hope): background then becomes central at the end
R3_TOMORROW  = [(0,1.5),(20,3.5),(38,5.5),(55,7.0),(68,7.8),(80,8.5),(90,9.0),(100,9.0)]
# 南方 — The Old South: fades over time
R4_SOUTH     = [(0,2.5),(18,5.5),(30,7.5),(48,7.5),(62,6.5),(75,5.5),(88,5.0),(100,5.0)]

# Pattern depths — what forms constellations
P0_SCORPIUS  = [(0,1.5),(18,5.0),(32,8.0),(48,9.0),(65,9.5),(80,9.5),(100,9.5)]
P1_LEO       = [(0,1.5),(15,4.0),(30,7.5),(50,8.0),(68,7.0),(80,6.5),(100,5.5)]
P2_CASSIOPEIA= [(0,1.0),(20,3.5),(38,7.0),(55,8.0),(70,8.5),(85,8.5),(100,8.5)]
P3_CAPRICORN = [(0,1.0),(25,3.5),(42,7.0),(60,8.0),(75,7.5),(90,8.0),(100,8.0)]

# ── Ecology element intensity curves (new in v28) ────────────────
E_MELANIE = [(0,0),(12,0),(18,5.5),(30,7.8),(55,8.5),(70,7.5),(78,0)]
E_BONNIE  = [(0,0),(55,0),(60,7.0),(65,9.2),(72,8.5),(76,0)]
E_CHARLES = [(0,0),(8,0),(11,5.0),(14,5.5),(18,0)]
E_FRANK   = [(0,0),(30,0),(34,5.5),(40,6.8),(46,5.5),(50,0)]
E_SELF    = [(0,0),(85,0),(90,5.0),(96,7.8),(100,8.5)]
E_YOUNG   = [(0,4.5),(8,8.2),(16,7.5),(25,0)]   # constellation depth
E_AWAKEN  = [(78,0),(83,5.2),(88,7.8),(100,7.8)] # constellation depth
E_WARFIRE = [(0,0),(16,0),(22,6.5),(28,8.8),(36,9.2),(42,7.5),(45,0)]
E_LONELY  = [(0,0),(82,0),(86,5.0),(90,7.2),(96,8.2),(100,8.8)]

def epoch_intensities(pct):
    return {
        'domain_i': [lerp(D0_PASSION,pct),lerp(D1_SURVIVAL,pct),lerp(D2_CHARM,pct),
                     lerp(D3_COLD_CALC,pct),lerp(D4_LONELINESS,pct)],
        'rel_i':    [lerp(R0_TARA,pct),lerp(R1_RHETT,pct),lerp(R2_ASHLEY,pct),
                     lerp(R3_TOMORROW,pct),lerp(R4_SOUTH,pct)],
        'pat_d':    [lerp(P0_SCORPIUS,pct),lerp(P1_LEO,pct),
                     lerp(P2_CASSIOPEIA,pct),lerp(P3_CAPRICORN,pct)],
    }

# Story beat descriptions
STORY_BEATS = [
    ( 0, '序幕', '佐治亚州的夏日·一切将开始'),
    ( 7, '初见', '烽火前夕·艾希礼的宣告'),
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

def get_beat(pct):
    best = min(STORY_BEATS, key=lambda b: abs(b[0]-pct))
    return best[1], best[2]

# ══════════════════════════════════════════════════════════════════
# UNIVERSE DATA  (positions fixed for all frames)
# ══════════════════════════════════════════════════════════════════
SCARLETT_UNIVERSE = {
    'name':'郝思嘉', 'title':'Scarlett_OHara',
    'domains':[
        # 0 热烈渴望 (pos: center, warm red)
        {'cx':12.35,'cy':11.70,'rx':8.06,'ry':5.33,'ang': 15,
         'dk':'#1a0003','mk':'#8c0022','bk':'#f03055','seed':11,'name':'热烈渴望'},
        # 1 生存本能 (pos: lower-right, warm orange)
        {'cx':18.85,'cy': 8.45,'rx':5.46,'ry':3.64,'ang':-35,
         'dk':'#190800','mk':'#8a4010','bk':'#f07028','seed':22,'name':'生存本能'},
        # 2 蛊惑魅力 (pos: lower-left, warm purple)
        {'cx': 6.24,'cy': 5.20,'rx':3.90,'ry':2.34,'ang': 70,
         'dk':'#160012','mk':'#6a0070','bk':'#e058e0','seed':55,'name':'蛊惑魅力'},
        # 3 冷静算计 (pos: upper-LEFT, cool teal)  ← Ashley orbits here
        {'cx': 5.20,'cy':18.85,'rx':4.94,'ry':3.12,'ang': 55,
         'dk':'#001418','mk':'#005058','bk':'#00c8a0','seed':33,'name':'冷静算计'},
        # 4 深层孤独 (pos: upper-right, cool blue)
        {'cx':22.75,'cy':17.55,'rx':4.42,'ry':2.60,'ang':-20,
         'dk':'#04000e','mk':'#180068','bk':'#4018d0','seed':44,'name':'深层孤独'},
    ],
    'relations':[
        # 0 塔拉  ← domain 0 (热烈渴望, center)
        {'name':'塔拉',  'domain':0,'angle':215,'color':'#ff4520'},
        # 1 瑞德  ← domain 2 (蛊惑魅力, lower-left)
        {'name':'瑞德',  'domain':2,'angle': 20,'color':'#ffb060'},
        # 2 艾希礼 ← domain 3 (冷静算计, UPPER-LEFT) ← THE LIFECYCLE STAR
        {'name':'艾希礼','domain':3,'angle':310,'color':'#a8c4ff'},
        # 3 明日  ← domain 4 (深层孤独, upper-right)
        {'name':'明日',  'domain':4,'angle':195,'color':'#cce0ff'},
        # 4 南方  ← domain 1 (生存本能, lower-right)
        {'name':'南方',  'domain':1,'angle':135,'color':'#fff4d0'},
    ],
    'patterns':[
        {'tmpl':'scorpius', 'name_cn':'天蝎座','cx':13.00,'cy':21.45,'sx':6.50,'sy':7.15,
         'soul':{'Antares':('热烈渴望','#ff6040'),'Epsilon':('生存意志','#ffb060'),'Shaula':('占有欲望','#c0d4ff')}},
        {'tmpl':'leo',      'name_cn':'狮子座','cx': 8.45,'cy':14.95,'sx':5.46,'sy':4.94,
         'soul':{'Regulus':('塔拉骄傲','#d8ecff'),'Denebola':('冷静算计','#a0c8ff')}},
        {'tmpl':'cassiopeia','name_cn':'仙后座','cx':20.15,'cy':13.65,'sx':4.55,'sy':2.60,
         'soul':{'Schedar':('不灭之志','#ffb060'),'Gamma':('深层孤独','#c0d8ff')}},
        {'tmpl':'capricorn', 'name_cn':'摩羯座','cx':17.55,'cy': 5.85,'sx':5.85,'sy':4.55,
         'soul':{'Algedi':('操纵心计','#f0e0a0'),'Deneb_Alg':('冷酷现实','#a0b8ff')}},
    ],
}

# ══════════════════════════════════════════════════════════════════
# PRE-COMPUTE ALL FIXED POSITIONS  (never change across frames)
# ══════════════════════════════════════════════════════════════════
def precompute_positions(universe):
    """All star/proto positions computed from BASE (unscaled) domain radii."""
    pos = {}
    for r in universe['relations']:
        d = universe['domains'][r['domain']]
        base_dist = max(d['rx'], d['ry']) * 0.82
        rad = np.radians(r['angle'])
        x = float(np.clip(d['cx']+np.cos(rad)*base_dist, 1.0, W-1.0))
        y = float(np.clip(d['cy']+np.sin(rad)*base_dist, 1.0, H-1.0))
        pos[r['name']] = (x, y)
        # Also compute proto positions (closer to domain center)
        proto_dist = max(d['rx'], d['ry']) * 0.48
        px_ = float(np.clip(d['cx']+np.cos(rad)*proto_dist, 1.0, W-1.0))
        py_ = float(np.clip(d['cy']+np.sin(rad)*proto_dist, 1.0, H-1.0))
        pos[r['name']+'_proto'] = (px_, py_)
    return pos

FIXED_POS = precompute_positions(SCARLETT_UNIVERSE)

# ══════════════════════════════════════════════════════════════════
# ECOLOGY ELEMENTS  (new in v28)
# Elements that emerge in empty spaces and die, releasing them
# Fields: name, etype ('star'/'const'/'neb'), birth%, death%,
#         curve, excl_r — plus type-specific fields
# ══════════════════════════════════════════════════════════════════
ECOLOGY_ELEMENTS = [
    # Stars — transient figures
    {'name':'梅兰妮', 'etype':'star', 'birth':12, 'death':78,
     'curve':E_MELANIE, 'color':'#c8e890', 'excl_r':2.5},

    {'name':'邦妮',   'etype':'star', 'birth':55, 'death':76,
     'curve':E_BONNIE,  'color':'#f0b0c0', 'excl_r':2.5},

    {'name':'查尔斯', 'etype':'star', 'birth':8,  'death':18,
     'curve':E_CHARLES, 'color':'#b8b8d0', 'excl_r':2.5},

    {'name':'弗兰克', 'etype':'star', 'birth':30, 'death':50,
     'curve':E_FRANK,   'color':'#c8a070', 'excl_r':2.5},

    {'name':'自我',   'etype':'star', 'birth':85, 'death':100,
     'curve':E_SELF,    'color':'#f0f0e0', 'excl_r':2.5},

    # Constellations — life stage patterns
    {'name':'青春娱乐', 'etype':'const', 'birth':0,  'death':25,
     'curve':E_YOUNG,
     'name_cn':'欢愉', 'tmpl':'young_fun',
     'sx':3.2, 'sy':2.5,
     'soul':{'B':('时尚','#ffd080'),'D':('宴会','#ffc060')},
     'excl_r':3.8},

    {'name':'醒悟真我', 'etype':'const', 'birth':78, 'death':100,
     'curve':E_AWAKEN,
     'name_cn':'醒悟', 'tmpl':'crown',
     'sx':3.0, 'sy':2.5,
     'soul':{'E':('自知','#f0e8d0'),'B':('明日','#f8f4e8')},
     'excl_r':3.8},

    # Nebulae — temporal emotional domains  (excl_r >= max(rx,ry) to prevent visual overlap)
    {'name':'战火创伤', 'etype':'neb', 'birth':16, 'death':45,
     'curve':E_WARFIRE,
     'rx':4.0, 'ry':2.6, 'ang':28,
     'dk':'#180400', 'mk':'#880018', 'bk':'#ff4820',
     'seed':77, 'excl_r':4.5},

    {'name':'晚秋孤寂', 'etype':'neb', 'birth':82, 'death':100,
     'curve':E_LONELY,
     'rx':3.2, 'ry':2.0, 'ang':-25,
     'dk':'#030810', 'mk':'#152048', 'bk':'#2848a0',
     'seed':88, 'excl_r':3.8},
]

# ── Ecology spatial pre-simulation ───────────────────────────────
_ERNG = np.random.RandomState(1337)
MIN_ECO_GAP = 1.5   # minimum visual gap between ecology element borders (~51 px)

def _find_empty_pos(eco_occ, static_occ, excl_r, attempts=800):
    """
    Place new element with guaranteed separation from other live ecology elements.
    eco_occ   : [(x,y,r)] currently-alive ecology — hard: gap >= MIN_ECO_GAP required
    static_occ: [(x,y,r)] domains + fixed stars  — soft: prefer to avoid, never blocks
    """
    margin = excl_r + 0.8
    best_valid, bv_score = None, -1e9
    best_any,   ba_score = None, -1e9

    for _ in range(attempts):
        x = float(_ERNG.uniform(margin, W - margin))
        y = float(_ERNG.uniform(margin, H - margin))

        eco_gap = min(
            math.sqrt((x-ox)**2+(y-oy)**2) - excl_r - r_
            for ox,oy,r_ in eco_occ
        ) if eco_occ else 999.0

        static_gap = min(
            math.sqrt((x-ox)**2+(y-oy)**2) - excl_r - r_
            for ox,oy,r_ in static_occ
        ) if static_occ else 999.0

        # Score: strongly prefer good ecology spacing, secondarily avoid domain cores
        score = eco_gap * 3.0 + max(-3.0, static_gap) * 0.5

        if score > ba_score:
            ba_score, best_any = score, (x, y)
        if eco_gap >= MIN_ECO_GAP and score > bv_score:
            bv_score, best_valid = score, (x, y)

    return best_valid or best_any or (W/2, H/2)


def compute_canvas_size(ecology_elements, domains):
    """Print recommended canvas size based on total soul element history."""
    max_concurrent = max(
        sum(1 for el in ecology_elements if el['birth'] <= p < el['death'])
        for p in range(0, 101)
    )
    avg_r = sum(el['excl_r'] for el in ecology_elements) / max(1, len(ecology_elements))
    eco_area = math.pi * (avg_r + MIN_ECO_GAP * 0.5)**2 * max_concurrent
    domain_area = sum(math.pi * max(d['rx'], d['ry'])**2 * 0.7 for d in domains)
    total = (domain_area + eco_area) / 0.60
    recommended = max(W, math.sqrt(total))
    print(f"  Soul canvas: {W:.0f}×{H:.0f}  "
          f"(ideal: {recommended:.1f}×{recommended:.1f}, "
          f"peak concurrent ecology: {max_concurrent})")


def precompute_ecology_positions(universe, n_frames):
    """
    Simulate ecology frame by frame.  Returns dict: name → (cx, cy)
    Domains are SOFT blockers (0.65× radius) — ecology can appear near their edges,
    filling the whole canvas.  Ecology-ecology gap is HARD-enforced (MIN_ECO_GAP).
    """
    positions = {}
    compute_canvas_size(ECOLOGY_ELEMENTS, universe['domains'])

    # Static soft blockers: domain cores + fixed surface stars + constellation centres
    domain_occ = [
        (d['cx'], d['cy'], max(d['rx'], d['ry']) * 0.65)
        for d in universe['domains']
    ]
    star_occ = [
        (sx, sy, 2.8)
        for name, (sx, sy) in FIXED_POS.items()
        if not name.endswith('_proto')
    ]
    pat_occ = [
        (p['cx'], p['cy'], max(p['sx'], p['sy']) * 0.40)
        for p in universe['patterns']
    ]
    static_occ = domain_occ + star_occ + pat_occ

    for f in range(n_frames):
        pct      = f / max(n_frames-1, 1) * 100.0
        prev_pct = (f-1) / max(n_frames-1, 1) * 100.0 if f > 0 else -1.0

        # Currently-alive ecology positions (hard constraint)
        eco_occ = [
            (positions[el['name']][0], positions[el['name']][1], el['excl_r'])
            for el in ECOLOGY_ELEMENTS
            if el['name'] in positions and el['birth'] <= pct < el['death']
        ]

        for el in ECOLOGY_ELEMENTS:
            if el['name'] in positions:
                continue
            if el['birth'] <= pct and el['birth'] > prev_pct:
                pos = _find_empty_pos(eco_occ, static_occ, el['excl_r'])
                positions[el['name']] = pos
                eco_occ.append((pos[0], pos[1], el['excl_r']))

    return positions

_N_ANIM = 60
ECOLOGY_POSITIONS = precompute_ecology_positions(SCARLETT_UNIVERSE, _N_ANIM)
print(f"  Ecology positions assigned: {len(ECOLOGY_POSITIONS)}")
for el in ECOLOGY_ELEMENTS:
    if el['name'] in ECOLOGY_POSITIONS:
        x,y = ECOLOGY_POSITIONS[el['name']]
        print(f"    {el['name']:<8} ({x:.1f},{y:.1f})")

# ══════════════════════════════════════════════════════════════════
# BUILD CHAR FROM EPOCH
# ══════════════════════════════════════════════════════════════════
def build_char(raw):
    nebs, idx_map = [], {}
    for i, d in enumerate(raw['domains']):
        if d['intensity'] >= DOMAIN_THRESH:
            sc = _neb_sc(d['intensity'])
            nebs.append((d['cx'],d['cy'],d['rx']*sc,d['ry']*sc,
                         d['ang'],d['dk'],d['mk'],d['bk'],
                         _neb_am(d['intensity']),d['seed'],d['name']))
            idx_map[i] = len(nebs)-1
    surf = [(r['name'],idx_map[r['domain']],r['angle'],_rel_mg(r['influence']),r['color'])
            for r in raw['relations']
            if r['influence']>=REL_THRESH and r['domain'] in idx_map]
    surf.sort(key=lambda s:-s[3])
    sf = lambda d: 0.75+0.25*(d/10.0)
    consts = [dict(c,sx=c['sx']*sf(c['depth']),sy=c['sy']*sf(c['depth']))
              for c in raw['patterns'] if c['depth']>=PAT_THRESH]
    return {'name':raw['name'],'title':raw['title'],
            'nebs':nebs,'surf':surf,'constellations':consts}

def build_epoch(universe, intensities, ecology_pct=None):
    d_raw = [dict(d,intensity=i) for d,i in zip(universe['domains'],intensities['domain_i'])]
    r_raw = [dict(r,influence=v) for r,v in zip(universe['relations'],intensities['rel_i'])]
    p_raw = [dict(p,depth=dd)   for p,dd in zip(universe['patterns'],intensities['pat_d'])]
    raw = dict(universe,domains=d_raw,relations=r_raw,patterns=p_raw)
    char = build_char(raw)

    # Proto signals (below threshold, above minimum)
    char['proto_frag_d']  = [(d,i) for d,i in zip(universe['domains'],intensities['domain_i'])
                              if 1.5<=i<4.5]
    char['proto_cloud_d'] = [(d,i) for d,i in zip(universe['domains'],intensities['domain_i'])
                              if 4.5<=i<DOMAIN_THRESH]
    char['proto_frag_r']  = [(r,v) for r,v in zip(universe['relations'],intensities['rel_i'])
                              if 1.5<=v<4.5]
    char['proto_cloud_r'] = [(r,v) for r,v in zip(universe['relations'],intensities['rel_i'])
                              if 4.5<=v<REL_THRESH]
    char['all_domains']   = universe['domains']

    # ── Ecology elements (v28 addition) ──────────────────────────
    if ecology_pct is not None:
        for el in ECOLOGY_ELEMENTS:
            if el['etype'] != 'neb':
                continue
            i = lerp(el['curve'], ecology_pct)
            if i <= 0.5:
                continue
            pos = ECOLOGY_POSITIONS.get(el['name'])
            if pos is None:
                continue
            ex, ey = pos
            if i >= DOMAIN_THRESH:
                sc = _neb_sc(i)
                char['nebs'].append((ex, ey, el['rx']*sc, el['ry']*sc,
                                     el['ang'], el['dk'], el['mk'], el['bk'],
                                     _neb_am(i), el['seed'], el['name']))
            elif i >= 4.5:
                # proto cloud: reuse domain proto cloud logic
                char['proto_cloud_d'].append(
                    ({'cx':ex,'cy':ey,'rx':el['rx'],'ry':el['ry'],'bk':el['bk']}, i))
            elif i >= 1.5:
                char['proto_frag_d'].append(
                    ({'cx':ex,'cy':ey,'bk':el['bk']}, i))
    return char

# ══════════════════════════════════════════════════════════════════
# RENDER  (positions fixed via FIXED_POS — no repulsion, no drift)
# ══════════════════════════════════════════════════════════════════
def render_frame(char, fixed_pos, pct, frame_idx, output_path,
                 fig_sz=FIG_SZ, dpi=DPI, ecology_positions=None):
    p = P; px = int(fig_sz*dpi)
    fig,ax = plt.subplots(figsize=(fig_sz,fig_sz),dpi=dpi)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.set_xlim(0,W); ax.set_ylim(0,H)
    ax.set_aspect('equal'); ax.axis('off')
    plt.subplots_adjust(0,0,1,1)

    # ── Sky atmosphere (v28) ─────────────────────────────────────
    render_sky_atmosphere(ax, pct, fig_sz, dpi)

    # ── Background stars (fixed seed) ────────────────────────────
    rng=np.random.default_rng(7)
    n=int(p['bg_n'])
    bx=rng.uniform(0,W,n); by=rng.uniform(0,H,n)
    band=p['bg_band']*np.exp(-np.abs(by-(0.48*H+(bx-W*.5)*.12))**2/(2*(H*.20)**2))
    raw=rng.power(0.35,n); s_pt=(0.03+raw*p['bg_maxr'])**2*1.25
    rv=rng.random(n)
    aalp=np.clip((0.06+0.94*raw)*p['bg_alpha']*(1+band*0.6),0.02,0.72)
    BGPOPS=[('#d0d0e2',rv>=0.16),('#fff0c0',(rv>=0.06)&(rv<0.16)),
            ('#90b4ff',(rv>=0.015)&(rv<0.06)),('#ffb088',rv<0.015)]
    for ch,mask in BGPOPS:
        if mask.any():
            ax.scatter(bx[mask],by[mask],s=s_pt[mask],c=ch,
                      alpha=float(aalp[mask].mean()*.50),linewidths=0,zorder=2)

    rng_f=np.random.default_rng(391)
    n_f=350; fx=rng_f.uniform(0,W,n_f); fy=rng_f.uniform(0,H,n_f)
    fraw=rng_f.power(0.50,n_f); fsize=(0.8+fraw*3.2)**2*0.28
    falpha=0.20+fraw*0.36; frv=rng_f.random(n_f)
    FPOPS=[('#c8c8dc',frv>=0.20),('#f0e8b0',(frv>=0.06)&(frv<0.20)),
           ('#8aaeff',(frv>=0.012)&(frv<0.06)),('#ffaa70',frv<0.012)]
    for fc,fm in FPOPS:
        if fm.any():
            ax.scatter(fx[fm],fy[fm],s=fsize[fm],c=fc,
                      alpha=float(falpha[fm].mean()),linewidths=0,zorder=2.6)

    # ── Legacy stars — traces of what once mattered ───────────────
    draw_legacy_stars(ax, pct, fig_sz)

    # ── Vignette ─────────────────────────────────────────────────
    yy,xx=np.mgrid[0:px,0:px]
    r2=(xx/px*2-1)**2+(yy/px*2-1)**2
    vig=np.zeros((px,px,4),dtype=np.float32)
    vig[:,:,3]=np.clip((r2-0.15)*0.60,0,0.52).astype(np.float32)
    ax.imshow(vig,extent=[0,W,0,H],origin='lower',
              interpolation='bilinear',zorder=2.8,aspect='auto')

    # ── Proto-fragments (1.5–4.5): faint dots ─────────────────────
    all_domains = char['all_domains']
    for d,i in char['proto_frag_d']:
        draw_proto_fragment(ax,d['cx'],d['cy'],i,d['bk'],fig_sz,z=2.91)
    for r_def,v in char['proto_frag_r']:
        x,y = fixed_pos.get(r_def['name']+'_proto', (r_def.get('cx',10),r_def.get('cy',10)))
        draw_proto_fragment(ax,x,y,v,r_def['color'],fig_sz,z=2.93)

    # ── Proto-clouds (4.5–6.0): condensing glow ──────────────────
    for d,i in char['proto_cloud_d']:
        draw_proto_cloud(ax,d['cx'],d['cy'],d['rx'],d['ry'],i,d['bk'],fig_sz,z=2.97)
    for r_def,v in char['proto_cloud_r']:
        x,y = fixed_pos.get(r_def['name']+'_proto', (10,10))
        frac=(v-4.5)/3.0; r_=P['sf_outer_r']*0.20*frac; alpha=frac*0.62
        draw_glow(ax,x,y,r_*4.0,r_def['color'],alpha*0.28,fig_sz,z=2.98)
        draw_star_small(ax,x,y,r_,r_def['color'],alpha*0.55,fig_sz,z=2.99)

    # ── Full nebulae (≥6.0): fractal clouds ──────────────────────
    if char['nebs']:
        neb=build_fractal_nebs(p,px,char['nebs'])
        ax.imshow(neb,extent=[0,W,0,H],origin='lower',
                  interpolation='bilinear',zorder=3,aspect='auto')
        for nd in char['nebs']:
            cx_d,cy_d,mk_col,lbl=nd[0],nd[1],nd[6],nd[10]
            ax.text(cx_d,cy_d,lbl,fontsize=11.0,color='#e8f4ff',alpha=0.68,
                   ha='center',va='center',fontstyle='italic',fontweight='light',zorder=3.5,
                   path_effects=[pe.withStroke(linewidth=5.0,foreground=mk_col)])

    # ── Bright bg stars pierce nebulae ────────────────────────────
    thr=np.percentile(raw,82)
    for ch,pm in BGPOPS:
        m=(raw>=thr)&pm
        if m.any():
            ax.scatter(bx[m],by[m],s=s_pt[m]*2.0,c=ch,
                      alpha=float(aalp[m].mean()*1.5),linewidths=0,zorder=6)

    # ── Constellations (FIXED cx,cy — no repulsion) ───────────────
    soul_r=p['cn_soul_r']; reg_r=soul_r*0.60
    for const in char['constellations']:
        cx_c,cy_c = const['cx'],const['cy']   # FIXED, never repelled
        csx=const['sx']; csy=const['sy']
        tmpl=TMPL[const['tmpl']]; soul=const.get('soul',{})
        coords={name:(cx_c+(nx-0.5)*csx,cy_c+(ny-0.5)*csy)
                for name,((nx,ny),*_) in tmpl['stars'].items()}
        for sa,sb in tmpl['lines']:
            x1,y1=coords[sa]; x2,y2=coords[sb]
            ax.plot([x1,x2],[y1,y2],'-',color='#8090c8',alpha=p['cl_alpha'],
                   lw=1.0,solid_capstyle='round',zorder=4)
        cst_placed=[]
        for name,((nx,ny),mag,spec_col) in tmpl['stars'].items():
            sx_c,sy_c=coords[name]
            is_soul=name in soul
            if is_soul:
                lbl_txt,soul_col=soul[name]
                draw_star_small(ax,sx_c,sy_c,soul_r,soul_col,p['cn_soul_a'],fig_sz,z=5)
                off=soul_r*3.5+0.10
                tx,ty,ha,va=find_label_pos(sx_c,sy_c,off,7.5,len(lbl_txt),fig_sz,cst_placed)
                ax.text(tx,ty,lbl_txt,fontsize=7.5,color=soul_col,alpha=0.85,
                       ha=ha,va=va,fontstyle='italic',fontweight='light',zorder=5,
                       path_effects=[pe.withStroke(linewidth=2.5,foreground='#080808')])
            else:
                draw_star_small(ax,sx_c,sy_c,reg_r,spec_col,p['cn_reg_a']*0.80,fig_sz,z=4.8)
                sr=reg_r*2.8*0.5
                cst_placed.append((sx_c-sr,sy_c-sr,sx_c+sr,sy_c+sr))
        all_x=[v[0] for v in coords.values()]; all_y=[v[1] for v in coords.values()]
        lbl_cx=sum(all_x)/len(all_x); lbl_cy=sum(all_y)/len(all_y)
        tx_n,ty_n,ha_n,va_n=find_label_pos(lbl_cx,lbl_cy,soul_r*2.5+0.15,
                                            9.0,len(const['name_cn']),fig_sz,cst_placed)
        ax.text(tx_n,ty_n,const['name_cn'],fontsize=9.0,color='#8090c8',alpha=0.65,
               ha=ha_n,va=va_n,fontweight='light',zorder=4.5,
               path_effects=[pe.withStroke(linewidth=2.5,foreground='#000406')])

    # ── Surface stars (FIXED positions from FIXED_POS) ────────────
    sf_r=p['sf_outer_r']
    for lbl,neb_idx,angle_deg,mag,col in char['surf']:
        sx_c,sy_c = fixed_pos[lbl]   # ← ALWAYS same position
        sc=np.clip((mag-6.0)/3.5,0.0,1.0); sc_p=sc**1.8
        r=sf_r*(0.18+0.82*sc_p); lsz=p['sf_lsz']*(0.48+0.52*sc_p)
        draw_star_full(ax,sx_c,sy_c,r,col,p['sf_alpha'],fig_sz,
                      glow_mult=p['sf_glow'],ray_mult=p['sf_ray'],z=7)
        ty=sy_c-r*p['sf_glow']*0.38-0.08
        ax.text(sx_c,ty,lbl,fontsize=lsz,color=col,alpha=0.92,
               ha='center',va='top',zorder=10,fontweight='light',
               path_effects=[pe.withStroke(linewidth=1.8,foreground='#060606')])

    # ── Ecology stars (v28) ──────────────────────────────────────
    for el in ECOLOGY_ELEMENTS:
        if el['etype'] != 'star':
            continue
        i = lerp(el['curve'], pct)
        if i <= 0.5:
            continue
        pos = ECOLOGY_POSITIONS.get(el['name'])
        if pos is None:
            continue
        ex, ey = pos
        sc = np.clip((i - 6.0)/3.5, 0.0, 1.0); sc_p = sc**1.8
        if i >= REL_THRESH:
            r  = p['sf_outer_r'] * (0.18 + 0.82*sc_p)
            lsz = p['sf_lsz'] * (0.48 + 0.52*sc_p)
            draw_star_full(ax, ex, ey, r, el['color'], p['sf_alpha'], fig_sz,
                           glow_mult=p['sf_glow'], ray_mult=p['sf_ray'], z=7)
            ty = ey - r*p['sf_glow']*0.38 - 0.08
            ax.text(ex, ty, el['name'], fontsize=lsz, color=el['color'],
                    alpha=0.88, ha='center', va='top', zorder=10,
                    fontweight='light',
                    path_effects=[pe.withStroke(linewidth=1.8,foreground='#060606')])
        elif i >= 4.5:
            frac = (i-4.5)/3.0
            r_ = p['sf_outer_r']*0.20*frac
            draw_glow(ax, ex, ey, r_*4.0, el['color'], frac*0.28, fig_sz, z=2.98)
            draw_star_small(ax, ex, ey, r_, el['color'], frac*0.55, fig_sz, z=2.99)
        elif i >= 1.5:
            draw_proto_fragment(ax, ex, ey, i, el['color'], fig_sz, z=2.93)

    # ── Ecology constellations (v28) ──────────────────────────────
    soul_r = p['cn_soul_r']; reg_r = soul_r*0.60
    for el in ECOLOGY_ELEMENTS:
        if el['etype'] != 'const':
            continue
        depth = lerp(el['curve'], pct)
        if depth < PAT_THRESH:
            continue
        pos = ECOLOGY_POSITIONS.get(el['name'])
        if pos is None:
            continue
        cx_e, cy_e = pos
        tmpl  = TMPL[el['tmpl']]
        soul  = el.get('soul', {})
        csx, csy = el['sx'], el['sy']
        sf_ = 0.75 + 0.25*(depth/10.0)
        csx *= sf_; csy *= sf_
        coords = {name: (cx_e+(nx-0.5)*csx, cy_e+(ny-0.5)*csy)
                  for name,((nx,ny),*_) in tmpl['stars'].items()}
        for sa,sb in tmpl['lines']:
            x1,y1=coords[sa]; x2,y2=coords[sb]
            ax.plot([x1,x2],[y1,y2],'-',color='#8090c8',
                    alpha=p['cl_alpha'], lw=1.0, solid_capstyle='round', zorder=4)
        cst_placed = []
        for sname,((nx,ny),mag,spec_col) in tmpl['stars'].items():
            sx_e, sy_e = coords[sname]
            is_soul = sname in soul
            if is_soul:
                lbl_txt, soul_col = soul[sname]
                draw_star_small(ax, sx_e, sy_e, soul_r, soul_col,
                                p['cn_soul_a'], fig_sz, z=5)
                off = soul_r*3.5+0.10
                tx,ty,ha,va = find_label_pos(sx_e,sy_e,off,7.5,len(lbl_txt),fig_sz,cst_placed)
                ax.text(tx,ty,lbl_txt,fontsize=7.5,color=soul_col,alpha=0.85,
                        ha=ha,va=va,fontstyle='italic',fontweight='light',zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.5,foreground='#080808')])
            else:
                draw_star_small(ax, sx_e, sy_e, reg_r, spec_col,
                                p['cn_reg_a']*0.80, fig_sz, z=4.8)
                sr = reg_r*2.8*0.5
                cst_placed.append((sx_e-sr,sy_e-sr,sx_e+sr,sy_e+sr))
        all_x=[v[0] for v in coords.values()]; all_y=[v[1] for v in coords.values()]
        lbl_cx=sum(all_x)/len(all_x); lbl_cy=sum(all_y)/len(all_y)
        tx_n,ty_n,ha_n,va_n = find_label_pos(lbl_cx,lbl_cy,soul_r*2.5+0.15,
                                              9.0,len(el['name_cn']),fig_sz,cst_placed)
        ax.text(tx_n,ty_n,el['name_cn'],fontsize=9.0,color='#8090c8',alpha=0.65,
                ha=ha_n,va=va_n,fontweight='light',zorder=4.5,
                path_effects=[pe.withStroke(linewidth=2.5,foreground='#000406')])

    # ── Frame title & story beat ───────────────────────────────────
    beat_name, beat_desc = get_beat(pct)
    sentence_approx = round(frame_idx * 100)
    ax.text(W/2, 0.92, f"郝思嘉  灵魂星图",
           fontsize=11.5,color='#aabbd0',alpha=0.65,
           ha='center',va='bottom',fontstyle='italic',zorder=11,
           path_effects=[pe.withStroke(linewidth=3.0,foreground='#000000')])
    ax.text(W/2, 0.40,
           f"第{sentence_approx}句话  ·  【{beat_name}】{beat_desc}",
           fontsize=7.5,color='#7888a0',alpha=0.60,
           ha='center',va='bottom',zorder=11,fontweight='light',
           path_effects=[pe.withStroke(linewidth=2.5,foreground='#000000')])

    # ── Ashley intensity annotation (upper-left) ───────────────────
    ashley_v = lerp(R2_ASHLEY, pct)
    ashley_tier = ('恒星' if ashley_v>=REL_THRESH else
                   '原始星云' if ashley_v>=4.5 else
                   '碎片' if ashley_v>=1.5 else '消散')
    ax.text(0.4, H-0.5,
           f"艾希礼  {ashley_v:.1f}  [{ashley_tier}]",
           fontsize=6.5,color='#a8c4ff',alpha=0.55,
           ha='left',va='top',zorder=11,fontweight='light',
           path_effects=[pe.withStroke(linewidth=2.0,foreground='#000000')])

    # ── Progress bar ──────────────────────────────────────────────
    bar_w=W*0.70; bar_x=(W-bar_w)/2; bar_y=0.12
    ax.plot([bar_x,bar_x+bar_w],[bar_y,bar_y],'-',
           color='#303848',alpha=0.60,lw=1.5,zorder=11)
    ax.plot([bar_x,bar_x+bar_w*(pct/100)],[bar_y,bar_y],'-',
           color='#6080b8',alpha=0.70,lw=1.5,zorder=11)
    ax.scatter([bar_x+bar_w*(pct/100)],[bar_y],s=8,c=['#90b0e8'],
              alpha=0.80,linewidths=0,zorder=11)

    output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(output_path),dpi=dpi,bbox_inches='tight',
               facecolor='black',edgecolor='none')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════
def make_gif(frame_paths, output_path, duration_ms=300):
    """Assemble PNG frames into animated GIF at fixed 900×900 resolution."""
    SIZE = (900, 900)
    frames_pil = []
    for p_ in frame_paths:
        img = Image.open(str(p_)).convert('RGB').resize(SIZE, Image.LANCZOS)
        frames_pil.append(img)
    # Quantize to palette for GIF
    frames_q = [f.quantize(colors=256,dither=Image.Dither.FLOYDSTEINBERG)
                for f in frames_pil]
    frames_q[0].save(
        str(output_path),
        save_all=True,
        append_images=frames_q[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

# ══════════════════════════════════════════════════════════════════
def main():
    N = _N_ANIM   # 60 frames
    print("═"*62)
    print("  Soul Star Map  v28  — Full Lifecycle Animation + Ecology")
    print(f"  {N} frames × 100 sentences = 郝思嘉全程心路历程")
    print("  所有星体位置锁定 — 艾希礼始终在左上角")
    print("  艾希礼弧线: 碎片→原始云→恒星→顶峰→消散→碎片")
    print("  v28新增: 生态元素 + 天空大气层")
    print(f"  Output: {OUT}/")
    print("═"*62)

    # Verify Ashley fixed position
    ax_x, ax_y = FIXED_POS['艾希礼']
    print(f"\n  艾希礼固定位置: ({ax_x:.2f}, {ax_y:.2f})  "
          f"← 左{ax_x/W*100:.0f}%  上{ax_y/H*100:.0f}%")

    t0 = time.time()
    frame_paths = []

    for fi in range(N):
        pct = fi * 100.0 / (N-1)
        intensities = epoch_intensities(pct)
        char = build_epoch(SCARLETT_UNIVERSE, intensities, ecology_pct=pct)

        ashley_v = lerp(R2_ASHLEY, pct)
        beat_name, _ = get_beat(pct)
        tier = ('★恒星' if ashley_v>=REL_THRESH else
                '◉云' if ashley_v>=4.5 else '·碎片')
        eco_alive = sum(1 for el in ECOLOGY_ELEMENTS if lerp(el['curve'],pct)>3.0)

        out = OUT / f"frame_{fi:03d}_pct{round(pct):03d}.png"
        t1 = time.time()
        render_frame(char, FIXED_POS, pct, fi, out)
        elapsed = time.time()-t1

        print(f"  [{fi:02d}/{N-1}] {pct:5.1f}% | {beat_name:4s} | "
              f"neb={len(char['nebs'])} star={len(char['surf'])} eco={eco_alive} | "
              f"Ashley={ashley_v:.1f}{tier} | {elapsed:.1f}s")
        frame_paths.append(out)

    # GIF
    gif_path = OUT / "Scarlett_生态星图.gif"
    print(f"\n  合成GIF ({N}帧 × 300ms = {N*300/1000:.1f}秒)…")
    make_gif(frame_paths, gif_path, duration_ms=300)
    print(f"  → {gif_path.name}")
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    print(f"\n  艾希礼生命轨迹验证:")
    for pct_ in [0,14,30,50,68,80,100]:
        v = lerp(R2_ASHLEY, pct_)
        tier_ = ('★恒星' if v>=REL_THRESH else '◉云' if v>=4.5 else '·碎片')
        print(f"    {pct_:3d}%: {v:.1f} {tier_}")

if __name__=='__main__':
    main()
