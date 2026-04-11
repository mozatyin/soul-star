"""
soul_star._engine — Shared rendering engine for Soul Star Map animations.

All visual and computation functions extracted from soul_star_scarlett.py (v28).
Character-specific data (curves, universe, ecology elements) is passed as
explicit parameters — no module-level globals that reference character data.

W, H (canvas dimensions) are threaded as explicit function parameters.
P (visual params) and TMPL (constellation templates) are module constants
since they are identical across all characters.
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

plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS',
    'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ── Visual / rendering constants (same for every character) ──────────
DOMAIN_THRESH = 6.0
REL_THRESH    = 7.5
PAT_THRESH    = 7.0

P = {
    'bg_n':6880,   'bg_maxr':0.166, 'bg_alpha':0.510, 'bg_band':2.095,
    'neb_gamma':2.902, 'neb_amax':0.817,  'neb_blur_k':0.110,
    'neb_amult':13.653,   'neb_n_oct':4.474,  'neb_env_n':0.481,
    'cl_alpha':0.498,
    'sf_outer_r':0.420, 'sf_alpha':1.000, 'sf_lsz':5.5,
    'sf_glow':11.01, 'sf_ray':9.497,
    'cn_soul_r':0.26, 'cn_soul_a':0.960, 'cn_reg_a':0.900,
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

# ══════════════════════════════════════════════════════════════════
# UTILITY
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

def hex2rgb(h): return tuple(int(h[i:i+2],16)/255.0 for i in (1,3,5))
def _neb_sc(v):  return (v / 10.0) ** 0.60
def _neb_am(v):  return (v / 10.0) ** 0.80 * 0.95
def _rel_mg(v):  return round(min(9.5, max(7.0, 7.5 + (v - 7.5) * 0.87)), 1)

def _pts(fs, W): return (fs/W)*72.0
def _s(r, fs, W): return max(0.1,(r*_pts(fs, W)*2.0)**2)

# ══════════════════════════════════════════════════════════════════
# VISUAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def draw_glow(ax, x, y, r, col, alpha, fs, z, W):
    for frac,al in [(1.00,0.06),(0.55,0.18),(0.25,0.38)]:
        ax.scatter(x,y,s=_s(r*frac,fs,W),c=[col],alpha=float(al*alpha),linewidths=0,zorder=z)

def draw_four_point(ax, x, y, outer_r, col, alpha, z):
    ir=outer_r*0.27; rgb=hex2rgb(col); v=[]
    for i in range(4):
        ao=np.pi/2-i*np.pi/2; ai=ao-np.pi/4
        v.append((x+np.cos(ao)*outer_r,y+np.sin(ao)*outer_r))
        v.append((x+np.cos(ai)*ir,y+np.sin(ai)*ir))
    ax.add_patch(MplPoly(v,closed=True,facecolor=(*rgb,float(alpha)),linewidth=0,zorder=z))

def draw_six_ray(ax, x, y, rlen, col, alpha, z):
    for i in range(6):
        ang=i*np.pi/3-np.pi/6; lw=1.0 if i%2==0 else 0.45
        for t0,t1,am in [(0.0,0.22,0.55),(0.22,0.55,0.18),(0.55,1.0,0.05)]:
            ax.plot([x+np.cos(ang)*rlen*t0,x+np.cos(ang)*rlen*t1],
                    [y+np.sin(ang)*rlen*t0,y+np.sin(ang)*rlen*t1],
                    '-',color=col,alpha=float(am*alpha),lw=lw,
                    solid_capstyle='round',zorder=z)

def draw_star_full(ax, x, y, outer_r, col, alpha, fs, glow_mult=7.5, ray_mult=9.5, z=7, W=26.0):
    draw_glow(ax,x,y,outer_r*glow_mult,col,alpha,fs,z-0.2,W)
    draw_six_ray(ax,x,y,outer_r*ray_mult,col,alpha,z)
    draw_glow(ax,x,y,outer_r*2.0,col,alpha*0.88,fs,z,W)
    draw_four_point(ax,x,y,outer_r,col,alpha*0.93,z+0.1)
    ax.scatter(x,y,s=_s(outer_r*0.22,fs,W),c=['#FFFDF5'],alpha=0.98,linewidths=0,zorder=z+0.2)

def draw_star_small(ax, x, y, outer_r, col, alpha, fs, z=5, W=26.0):
    draw_glow(ax,x,y,outer_r*3.0,col,alpha*0.50,fs,z-0.1,W)
    draw_six_ray(ax,x,y,outer_r*4.5,col,alpha*0.65,z)
    draw_four_point(ax,x,y,outer_r,col,alpha*0.90,z+0.1)
    ax.scatter(x,y,s=_s(outer_r*0.25,fs,W),c=['#FFFDF5'],alpha=0.95,linewidths=0,zorder=z+0.2)

def draw_proto_fragment(ax, x, y, intensity, color, fs, z=2.91, W=26.0):
    frac=max(0,(intensity-1.5)/3.0)
    r=P['sf_outer_r']*0.10*(0.3+0.7*frac)
    draw_glow(ax,x,y,r*4.0,color,frac*0.22,fs,z,W)
    ax.scatter(x,y,s=_s(r*0.5,fs,W),c=[color],alpha=float(frac*0.55),linewidths=0,zorder=z+0.1)

def draw_proto_cloud(ax, x, y, rx, ry, intensity, color_hot, fs, z=2.97, W=26.0):
    frac=max(0,(intensity-4.5)/1.5)
    r=max(rx,ry)*_neb_sc(intensity)*0.42
    for f,a in [(1.0,0.04),(0.6,0.10),(0.3,frac*0.20)]:
        ax.scatter(x,y,s=_s(r*f,fs,W)*3.0,c=[color_hot],alpha=float(a),linewidths=0,zorder=z)

def draw_legacy_stars(ax, pct, fig_sz, W, ecology_elements, ecology_positions):
    """Dead ecology elements leave faint memorial stars — once important, now a quiet trace."""
    for el in ecology_elements:
        if el['death'] > pct:
            continue
        pos = ecology_positions.get(el['name'])
        if pos is None:
            continue
        ex, ey = pos
        col  = el.get('color', el.get('bk', '#ffffff'))  # star→color, neb→bk
        age  = min(1.0, (pct - el['death']) / 25.0)
        alp  = max(0.12, 0.38 - 0.18*age)
        r    = max(0.030, 0.068 - 0.025*age)
        draw_glow(ax, ex, ey, r*3.5, col, alp*0.42, fig_sz, z=2.70, W=W)
        ax.scatter([ex],[ey], s=_s(r,      fig_sz, W), c=[col],
                  alpha=float(alp),      linewidths=0, zorder=2.72)
        ax.scatter([ex],[ey], s=_s(r*0.42, fig_sz, W), c=['white'],
                  alpha=float(alp*0.55), linewidths=0, zorder=2.73)

# ── LABEL PLACEMENT ───────────────────────────────────────────────
def find_label_pos(cx, cy, offset, lsz, nchars, fs, placed, W, H):
    pts=_pts(fs, W); lw=nchars*lsz/pts*0.62; lh=lsz/pts*1.25
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
def build_fractal_nebs(p, px, nebs, W, H):
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
# SKY PHENOMENA — rare atmospheric/astronomical events
# Probabilities tuned for "watch 3-5 min/day" session cadence.
# Natural frequency brackets → scaled probability per video play:
#   >1/5min (common)   → 1x  → 60-80% per play
#   ~1/hour            → 10x → 20-35%
#   ~1/year            → 100x → 12-20%
#   ~1/decade          → 1000x capped → 5-8%
#   ~1/century         → 1000x capped → 2-3%
#   never (gift)       → 2%
# ══════════════════════════════════════════════════════════════════
SKY_EVENTS_CONFIG = {
    # type              p/play  max  dur_min  dur_max
    'meteor':          (0.65,   2,    3,       6   ),
    'satellite':       (0.30,   1,    50,      80  ),
    'aurora_surge':    (0.18,   1,    20,      35  ),
    'variable_star':   (0.22,   1,    -1,      -1  ),  # whole video
    'comet':           (0.07,   1,    60,      90  ),
    'supernova':       (0.03,   1,    30,      50  ),
    'nebula_breath':   (1.00,   1,    -1,      -1  ),  # always
    'dark_cloud':      (0.15,   1,    40,      60  ),
    'globular_cluster':(0.12,   1,    25,      40  ),
    'grav_ripple':     (0.02,   1,    20,      30  ),
}


def precompute_sky_events(n_frames, px, rng_seed=777, story_beats=None, sky_event_beats=None, guarantee_all=True):
    """Pre-compute sky event schedule for a full video render.

    sky_event_beats: {event_type: {beat_name: boost_multiplier}}
    When a matching story beat is active, probability is boosted and
    event timing is biased toward that beat window.
    """
    rng = np.random.default_rng(rng_seed)
    events = []

    def _frame_weights(dur, beat_boosts):
        w = np.ones(n_frames, dtype=np.float64)
        if not story_beats or not beat_boosts:
            w[max(0, n_frames - dur):] = 0.0
            total = w.sum()
            return w / total if total > 1e-9 else np.ones(n_frames) / n_frames
        current = story_beats[0][1]
        for fi in range(n_frames):
            pct = fi * 100.0 / max(n_frames - 1, 1)
            for bp, bn, _ in story_beats:
                if pct >= bp:
                    current = bn
            if current in beat_boosts:
                w[fi] *= beat_boosts[current]
        w[max(0, n_frames - dur):] = 0.0
        total = w.sum()
        return w / total if total > 1e-9 else np.ones(n_frames) / n_frames

    def _effective_p(base_p, etype):
        """Boost base probability when matching beats exist in this video."""
        if not sky_event_beats or etype not in sky_event_beats:
            return base_p
        if not story_beats:
            return base_p
        beat_boosts = sky_event_beats[etype]
        # Check if any matching beat exists in this video's story
        beat_names_in_video = {bn for _, bn, _ in story_beats}
        max_boost = max((v for k, v in beat_boosts.items() if k in beat_names_in_video), default=1.0)
        if max_boost <= 1.0:
            return base_p
        # Logarithmic compression: high boosts → meaningful but not overwhelming increase
        import math as _math
        boosted = base_p * (1.0 + _math.log(max_boost) * 0.6)
        return min(0.95, boosted)

    def _sched(etype, dur):
        """Sample a start frame using story beat weights."""
        beats = sky_event_beats.get(etype, {}) if sky_event_beats else {}
        w = _frame_weights(dur, beats)
        return int(rng.choice(n_frames, p=w))

    def _rand_edge():
        edge = int(rng.integers(0, 4))
        if edge == 0: return int(rng.integers(0, px)), 0
        if edge == 1: return px-1, int(rng.integers(0, px))
        if edge == 2: return int(rng.integers(0, px)), px-1
        return 0, int(rng.integers(0, px))

    # nebula_breath: always present, whole video
    events.append({'type':'nebula_breath','start_frame':0,'end_frame':n_frames-1,
                   'params':{'amplitude':0.12}})

    # variable_star: whole video if triggered
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['variable_star']
    ep = _effective_p(p0, 'variable_star')
    if rng.random() < ep:
        events.append({'type':'variable_star','start_frame':0,'end_frame':n_frames-1,
                       'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                 'cy':int(rng.integers(px//6,5*px//6)),
                                 'period_frames': n_frames*2,
                                 'amplitude':0.55}})

    # meteor: up to 2
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['meteor']
    ep = _effective_p(p0, 'meteor')
    for _ in range(mx):
        if rng.random() < ep / mx:
            dur = int(rng.integers(dmin, dmax+1))
            start = _sched('meteor', dur)
            x0, y0 = _rand_edge()
            angle = float(rng.uniform(0.1, math.pi - 0.1))
            length = int(rng.integers(px//8, px//3))
            x1 = int(np.clip(x0 + length*math.cos(angle), 0, px-1))
            y1 = int(np.clip(y0 + length*math.sin(angle), 0, px-1))
            events.append({'type':'meteor','start_frame':start,'end_frame':start+dur-1,
                           'params':{'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                                     'r':float(rng.uniform(0.88,1.0)),
                                     'g':float(rng.uniform(0.88,1.0)),
                                     'b':float(rng.uniform(0.92,1.0))}})

    # satellite
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['satellite']
    ep = _effective_p(p0, 'satellite')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('satellite', dur)
        y = int(rng.integers(px//8, 7*px//8))
        going_right = bool(rng.random() > 0.5)
        x0, x1 = (0, px-1) if going_right else (px-1, 0)
        y1 = y + int(rng.integers(-px//15, px//15))
        events.append({'type':'satellite','start_frame':start,'end_frame':start+dur-1,
                       'params':{'x0':x0,'y0':y,'x1':x1,'y1':int(np.clip(y1,0,px-1))}})

    # aurora_surge
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['aurora_surge']
    ep = _effective_p(p0, 'aurora_surge')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('aurora_surge', dur)
        events.append({'type':'aurora_surge','start_frame':start,'end_frame':start+dur-1,
                       'params':{'peak_mult':float(rng.uniform(2.5, 4.5))}})

    # comet
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['comet']
    ep = _effective_p(p0, 'comet')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('comet', dur)
        y0 = int(rng.integers(px//5, 4*px//5))
        y1 = y0 + int(rng.integers(-px//10, px//10))
        events.append({'type':'comet','start_frame':start,'end_frame':start+dur-1,
                       'params':{'x0':0,'y0':y0,'x1':px*2//5,'y1':int(np.clip(y1,0,px-1)),
                                 'tail_len':int(rng.integers(30,60))}})

    # supernova
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['supernova']
    ep = _effective_p(p0, 'supernova')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('supernova', dur)
        events.append({'type':'supernova','start_frame':start,'end_frame':start+dur-1,
                       'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                 'cy':int(rng.integers(px//6,5*px//6))}})

    # dark_cloud
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['dark_cloud']
    ep = _effective_p(p0, 'dark_cloud')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('dark_cloud', dur)
        events.append({'type':'dark_cloud','start_frame':start,'end_frame':start+dur-1,
                       'params':{'cy':int(rng.integers(px//4,3*px//4)),
                                 'ry':int(rng.integers(px//8,px//4)),
                                 'rx':int(rng.integers(px//4,px//2)),
                                 'opacity':float(rng.uniform(0.25,0.45))}})

    # globular_cluster
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['globular_cluster']
    ep = _effective_p(p0, 'globular_cluster')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('globular_cluster', dur)
        events.append({'type':'globular_cluster','start_frame':start,'end_frame':start+dur-1,
                       'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                 'cy':int(rng.integers(px//6,5*px//6)),
                                 'radius':int(rng.integers(px//15,px//8)),
                                 'n_stars':int(rng.integers(60,120)),
                                 'seed':int(rng.integers(0,99999))}})

    # grav_ripple
    p0, mx, dmin, dmax = SKY_EVENTS_CONFIG['grav_ripple']
    ep = _effective_p(p0, 'grav_ripple')
    if rng.random() < ep:
        dur = int(rng.integers(dmin, dmax+1))
        start = _sched('grav_ripple', dur)
        events.append({'type':'grav_ripple','start_frame':start,'end_frame':start+dur-1,
                       'params':{'cx':int(rng.integers(px//4,3*px//4)),
                                 'cy':int(rng.integers(px//4,3*px//4)),
                                 'max_r':int(rng.integers(px//5,px//3))}})

    # ── Guarantee every event type appears at least once ─────────────
    if guarantee_all:
        scheduled_types = {e['type'] for e in events}

        for etype in ('meteor', 'satellite', 'aurora_surge', 'comet',
                      'supernova', 'dark_cloud', 'globular_cluster', 'grav_ripple'):
            if etype in scheduled_types:
                continue
            _p0, _mx, dmin_g, dmax_g = SKY_EVENTS_CONFIG[etype]
            dur = int(rng.integers(dmin_g, dmax_g + 1))
            start = _sched(etype, dur)

            if etype == 'meteor':
                x0, y0 = _rand_edge()
                angle = float(rng.uniform(0.1, math.pi - 0.1))
                length = int(rng.integers(px//8, px//3))
                x1 = int(np.clip(x0 + length*math.cos(angle), 0, px-1))
                y1 = int(np.clip(y0 + length*math.sin(angle), 0, px-1))
                events.append({'type':'meteor','start_frame':start,'end_frame':start+dur-1,
                               'params':{'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                                         'r':float(rng.uniform(0.88,1.0)),
                                         'g':float(rng.uniform(0.88,1.0)),
                                         'b':float(rng.uniform(0.92,1.0))}})
            elif etype == 'satellite':
                y = int(rng.integers(px//8, 7*px//8))
                going_right = bool(rng.random() > 0.5)
                x0g, x1g = (0, px-1) if going_right else (px-1, 0)
                y1 = int(np.clip(y + int(rng.integers(-px//15, px//15)), 0, px-1))
                events.append({'type':'satellite','start_frame':start,'end_frame':start+dur-1,
                               'params':{'x0':x0g,'y0':y,'x1':x1g,'y1':y1}})
            elif etype == 'aurora_surge':
                events.append({'type':'aurora_surge','start_frame':start,'end_frame':start+dur-1,
                               'params':{'peak_mult':float(rng.uniform(2.5, 4.5))}})
            elif etype == 'comet':
                y0g = int(rng.integers(px//5, 4*px//5))
                y1g = int(np.clip(y0g + int(rng.integers(-px//10, px//10)), 0, px-1))
                events.append({'type':'comet','start_frame':start,'end_frame':start+dur-1,
                               'params':{'x0':0,'y0':y0g,'x1':px*2//5,'y1':y1g,
                                         'tail_len':int(rng.integers(30,60))}})
            elif etype == 'supernova':
                events.append({'type':'supernova','start_frame':start,'end_frame':start+dur-1,
                               'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                         'cy':int(rng.integers(px//6,5*px//6))}})
            elif etype == 'dark_cloud':
                events.append({'type':'dark_cloud','start_frame':start,'end_frame':start+dur-1,
                               'params':{'cy':int(rng.integers(px//4,3*px//4)),
                                         'ry':int(rng.integers(px//8,px//4)),
                                         'rx':int(rng.integers(px//4,px//2)),
                                         'opacity':float(rng.uniform(0.25,0.45))}})
            elif etype == 'globular_cluster':
                events.append({'type':'globular_cluster','start_frame':start,'end_frame':start+dur-1,
                               'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                         'cy':int(rng.integers(px//6,5*px//6)),
                                         'radius':int(rng.integers(px//15,px//8)),
                                         'n_stars':int(rng.integers(60,120)),
                                         'seed':int(rng.integers(0,99999))}})
            elif etype == 'grav_ripple':
                events.append({'type':'grav_ripple','start_frame':start,'end_frame':start+dur-1,
                               'params':{'cx':int(rng.integers(px//4,3*px//4)),
                                         'cy':int(rng.integers(px//4,3*px//4)),
                                         'max_r':int(rng.integers(px//5,px//3))}})

        if 'variable_star' not in scheduled_types:
            events.append({'type':'variable_star','start_frame':0,'end_frame':n_frames-1,
                           'params':{'cx':int(rng.integers(px//6,5*px//6)),
                                     'cy':int(rng.integers(px//6,5*px//6)),
                                     'period_frames': n_frames*2,
                                     'amplitude':0.55}})

    return events


def _sky_line(img, px, x0, y0, x1, y1, r, g, b, brightness_arr):
    """Draw a line segment into img, brightness_arr has one value per point."""
    n = max(abs(x1-x0), abs(y1-y0), 1)
    xs = np.round(np.linspace(x0, x1, n+1)).astype(int)
    ys = np.round(np.linspace(y0, y1, n+1)).astype(int)
    valid = (xs >= 0) & (xs < px) & (ys >= 0) & (ys < px)
    bv = np.broadcast_to(brightness_arr, (n+1,)) if np.ndim(brightness_arr)==0 else brightness_arr
    img[ys[valid], xs[valid], 0] += r * bv[valid]
    img[ys[valid], xs[valid], 1] += g * bv[valid]
    img[ys[valid], xs[valid], 2] += b * bv[valid]


def render_sky_events_to_img(img, px, frame_idx, sky_events):
    """Render all active sky events into pixel buffer img (in-place, float32)."""
    if not sky_events:
        return

    # Pre-build meshgrid only if needed
    _need_grid = any(e['type'] in ('supernova','dark_cloud','grav_ripple','variable_star')
                     for e in sky_events
                     if e['start_frame'] <= frame_idx <= e['end_frame'])
    if _need_grid:
        yy, xx = np.mgrid[0:px, 0:px].astype(np.float32)

    for evt in sky_events:
        if not (evt['start_frame'] <= frame_idx <= evt['end_frame']):
            continue
        etype = evt['type']
        p     = evt['params']
        dur   = max(evt['end_frame'] - evt['start_frame'] + 1, 1)
        phase = (frame_idx - evt['start_frame']) / max(dur - 1, 1)  # 0→1

        # ── Meteor ────────────────────────────────────────────────
        if etype == 'meteor':
            # head moves along trajectory; trail fades behind
            hx = int(p['x0'] + (p['x1']-p['x0']) * phase)
            hy = int(p['y0'] + (p['y1']-p['y0']) * phase)
            tp = max(0.0, phase - 0.35)
            tx = int(p['x0'] + (p['x1']-p['x0']) * tp)
            ty = int(p['y0'] + (p['y1']-p['y0']) * tp)
            n  = max(abs(hx-tx), abs(hy-ty), 1)
            t_vals = np.linspace(0.0, 1.0, n+1, dtype=np.float32)  # 0=tail, 1=head
            fade = float((1.0 - phase) ** 0.5)   # slow fade: bright start → dim end
            brightness = t_vals**2 * fade * 2.8
            _sky_line(img, px, tx, ty, hx, hy, p['r'], p['g'], p['b'], brightness)

        # ── Satellite ─────────────────────────────────────────────
        elif etype == 'satellite':
            cx = int(p['x0'] + (p['x1']-p['x0']) * phase)
            cy = int(p['y0'] + (p['y1']-p['y0']) * phase)
            if 0 <= cx < px and 0 <= cy < px:
                img[cy, cx, 0] += 0.70
                img[cy, cx, 1] += 0.72
                img[cy, cx, 2] += 0.75
                # soft cross-pixel glow
                for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                    nx_, ny_ = cx+dx, cy+dy
                    if 0 <= nx_ < px and 0 <= ny_ < px:
                        img[ny_, nx_, :] += 0.18

        # ── Comet ─────────────────────────────────────────────────
        elif etype == 'comet':
            hx = int(p['x0'] + (p['x1']-p['x0']) * phase)
            hy = int(p['y0'] + (p['y1']-p['y0']) * phase)
            # Bright head (Moffat-like: just a bright pixel + glow)
            if 0 <= hx < px and 0 <= hy < px:
                img[hy, hx, :] += 1.2
                for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)):
                    nx_, ny_ = hx+dx, hy+dy
                    if 0 <= nx_ < px and 0 <= ny_ < px:
                        img[ny_, nx_, :] += 0.40
            # Tail: points OPPOSITE to direction of motion (anti-sunward)
            dx_dir = -(p['x1']-p['x0']); dy_dir = -(p['y1']-p['y0'])
            norm = math.sqrt(dx_dir**2 + dy_dir**2) + 1e-8
            dx_dir /= norm; dy_dir /= norm
            tail = p['tail_len']
            d_arr = np.arange(1, tail+1, dtype=np.float32)
            ivals = np.exp(-d_arr / (tail*0.35)) * 0.55
            xs_t = np.clip((hx + dx_dir*d_arr + 0.5).astype(int), 0, px-1)
            ys_t = np.clip((hy + dy_dir*d_arr + 0.5).astype(int), 0, px-1)
            img[ys_t, xs_t, 0] += ivals * 0.85
            img[ys_t, xs_t, 1] += ivals * 0.90
            img[ys_t, xs_t, 2] += ivals * 1.00

        # ── Supernova ─────────────────────────────────────────────
        elif etype == 'supernova':
            cx, cy = int(p['cx']), int(p['cy'])
            # Rise fast (first 15% of dur), then slow exponential decay
            if phase <= 0.15:
                intensity = phase / 0.15
            else:
                intensity = math.exp(-5.0 * (phase - 0.15))
            # Glow ring
            r_glow = int(intensity * px * 0.08)
            if r_glow > 0:
                dr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                glow = intensity * 1.8 * np.exp(-0.5 * (dr / max(r_glow*0.5, 1))**2)
                img[:,:,0] += glow.astype(np.float32) * 1.0
                img[:,:,1] += glow.astype(np.float32) * 0.85
                img[:,:,2] += glow.astype(np.float32) * 0.55
            # Bright core
            if 0 <= cx < px and 0 <= cy < px:
                img[cy, cx, 0] += intensity * 3.0
                img[cy, cx, 1] += intensity * 2.5
                img[cy, cx, 2] += intensity * 1.5

        # ── Dark cloud ────────────────────────────────────────────
        elif etype == 'dark_cloud':
            # Drifts in from left, peaks in middle, exits right
            cx_cloud = int((phase - 0.5) * px * 1.8 + px * 0.5)
            cy_cloud = int(p['cy'])
            dr = np.sqrt(((xx - cx_cloud)/max(p['rx'],1))**2 +
                         ((yy - cy_cloud)/max(p['ry'],1))**2)
            # Smooth bell: max opacity in center, fades at edges
            fade_in_out = math.sin(math.pi * phase)  # 0→1→0
            mask = np.exp(-dr**2 * 0.8) * p['opacity'] * fade_in_out
            # Darken the image
            img[:,:,0] *= (1.0 - mask).astype(np.float32)
            img[:,:,1] *= (1.0 - mask).astype(np.float32)
            img[:,:,2] *= (1.0 - mask).astype(np.float32)

        # ── Globular cluster ──────────────────────────────────────
        elif etype == 'globular_cluster':
            rng_gc = np.random.default_rng(p['seed'])
            fade = float(math.sin(math.pi * phase))  # fade in/out
            n_gc = int(p['n_stars'])
            # Gaussian distribution around center
            r_arr = rng_gc.exponential(p['radius']*0.4, n_gc).clip(0, p['radius'])
            ang_arr = rng_gc.uniform(0, 2*math.pi, n_gc)
            gx = (p['cx'] + r_arr*np.cos(ang_arr)).astype(int)
            gy = (p['cy'] + r_arr*np.sin(ang_arr)).astype(int)
            gm = rng_gc.uniform(0.06, 0.25, n_gc).astype(np.float32) * fade
            valid = (gx >= 0) & (gx < px) & (gy >= 0) & (gy < px)
            img[gy[valid], gx[valid], 0] += gm[valid] * 0.90
            img[gy[valid], gx[valid], 1] += gm[valid] * 0.92
            img[gy[valid], gx[valid], 2] += gm[valid] * 1.00

        # ── Gravitational ripple ──────────────────────────────────
        elif etype == 'grav_ripple':
            cx, cy = int(p['cx']), int(p['cy'])
            r_ring = phase * p['max_r']
            sigma  = max(2.0, p['max_r'] * 0.04)
            dr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            ring = np.exp(-0.5 * ((dr - r_ring) / sigma)**2)
            # Fades as it expands
            amp = float((1.0 - phase) * 0.18)
            img[:,:,0] += (ring * amp * 0.70).astype(np.float32)
            img[:,:,1] += (ring * amp * 0.80).astype(np.float32)
            img[:,:,2] += (ring * amp * 1.00).astype(np.float32)

        # ── Variable star (extra slow pulse, distinct from twinkling) ──
        elif etype == 'variable_star':
            cx, cy = int(p['cx']), int(p['cy'])
            pulse = float(0.5 + 0.5*math.sin(2*math.pi * frame_idx / p['period_frames']))
            brightness = p['amplitude'] * pulse
            dr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            glow = np.exp(-0.5*(dr/max(3.0, px*0.012))**2) * brightness
            img[:,:,0] += (glow * 0.95).astype(np.float32)
            img[:,:,1] += (glow * 0.88).astype(np.float32)
            img[:,:,2] += (glow * 0.72).astype(np.float32)
            # Bright core pixel
            if 0 <= cx < px and 0 <= cy < px:
                img[cy, cx, :] += brightness * 1.5


# ══════════════════════════════════════════════════════════════════
# SKY ATMOSPHERE
# ══════════════════════════════════════════════════════════════════

def render_sky_atmosphere(ax, pct, fig_sz, dpi, W, H, bg_temp_curve, aurora_curve, aurora_surge_mult=1.0):
    """Render atmospheric sky gradient + optional aurora. zorder=0.5."""
    temp     = lerp(bg_temp_curve, pct)
    aurora_s = lerp(aurora_curve, pct)

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
            contrib = aurora_s * aurora_surge_mult * strength * 0.52 * env[:,None] * wave[None,:]
            img += (contrib[:,:,None] * ac[None,None,:]).astype(np.float32)

    img = np.clip(img, 0, 1)
    ax.imshow(img, extent=[0, W, 0, H], origin='lower',
              aspect='auto', zorder=0.5, interpolation='bilinear')

# ══════════════════════════════════════════════════════════════════
# STARFIELD — Moffat PSF stamps (physically accurate, sharp stars)
# ══════════════════════════════════════════════════════════════════

def _moffat_stamp(size, alpha=0.85, beta=4.765):
    """
    Moffat PSF kernel, normalized so center pixel = 1.0.
    Moffat: PSF(r) = (1 + (r/alpha)^2)^(-beta)
    At r=1px with alpha=0.85: value = (1+1.385)^-4.765 ≈ 0.014  — nearly ALL
    light in center pixel. This is the "sparkle" quality of real stars.
    """
    c = size // 2
    yy, xx = np.ogrid[-c:c+1, -c:c+1]
    r = np.sqrt(xx.astype(np.float32)**2 + yy.astype(np.float32)**2)
    psf = (1.0 + (r / alpha)**2).astype(np.float32) ** (-beta)
    return (psf / float(psf[c, c])).astype(np.float32)   # center = 1.0

# Stamp library — 6 tiers: (mag_lo, mag_hi, stamp)
# Small alpha → sharp concentrated core; larger alpha → slightly extended bright star
_STAMP_BINS = [
    (0.000, 0.007, _moffat_stamp( 3, alpha=0.70)),   # sub-pixel dim — almost single-pixel
    (0.007, 0.035, _moffat_stamp( 5, alpha=0.80)),   # faint
    (0.035, 0.140, _moffat_stamp( 7, alpha=0.90)),   # medium-faint
    (0.140, 0.380, _moffat_stamp( 9, alpha=1.20)),   # medium
    (0.380, 0.750, _moffat_stamp(13, alpha=1.80)),   # bright — slight halo
    (0.750, 1.200, _moffat_stamp(19, alpha=3.00)),   # very bright — visible halo
]


def render_starfield_bg(ax, W, H, px, frame_idx, sky_events=None):
    """
    Photorealistic starfield via Moffat PSF stamps.

    Each star is rendered by "stamping" a Moffat PSF kernel at its pixel position,
    scaled by magnitude.  Moffat concentrates nearly all light in the center pixel
    with sharp falloff — producing the crisp "sparkle" of real astrophotography.

    No Gaussian blur smear. No blurry halos. Stars are sharp point sources.

    Background: deep navy sky baked into the image (zorder=1.5, solid).
    """
    # Deep navy sky base — matches real dark-sky astrophotography background
    # RGB(35,35,50)/255 ≈ (0.137,0.137,0.196) is the research-validated sky floor
    # We use a slightly darker version matching our render_sky_atmosphere output
    # ── Sky base — atmospheric gradient (not pure black) ─────────────────────
    # Real night sky: zenith darker/bluer, horizon subtly lighter with airglow
    rng_sky = np.random.default_rng(888)
    img = np.empty((px, px, 3), dtype=np.float32)
    # t=0 at bottom row (horizon), t=1 at top (zenith)
    t_vert = np.linspace(0.0, 1.0, px, dtype=np.float32)
    c_hor = np.array([0.022, 0.018, 0.092], dtype=np.float32)  # warm-blue horizon
    c_zen = np.array([0.007, 0.009, 0.065], dtype=np.float32)  # deep indigo zenith
    sky_grad = (c_hor[np.newaxis,:] * (1 - t_vert[:,np.newaxis]) +
                c_zen[np.newaxis,:] * t_vert[:,np.newaxis])     # (px,3)
    img[:] = sky_grad[:, np.newaxis, :]                         # broadcast → (px,px,3)
    # Subtle large-scale atmospheric variation (airglow patches)
    _sn = rng_sky.standard_normal((px//16, px//16)).astype(np.float32)
    _sn = gaussian_filter(_sn, sigma=3)
    _sn = (_sn - _sn.min()) / (_sn.max() - _sn.min() + 1e-8)
    _sn_pil = Image.fromarray((_sn*255).astype(np.uint8)).resize((px,px), Image.BILINEAR)
    _sn_full = np.array(_sn_pil, dtype=np.float32) / 255.0
    vs = 0.005  # variation strength — very subtle
    img[:,:,0] += _sn_full * vs * 0.5
    img[:,:,1] += _sn_full * vs * 0.6
    img[:,:,2] += _sn_full * vs * 1.2

    # ── Milky Way: subtle brightening of the galactic band ────────────
    # fBM noise at 1/8 scale → upscale → soft blur = diffuse cloud glow
    rng_mw = np.random.default_rng(777)
    mw_n = max(48, px // 8)
    xs_m = np.arange(mw_n, dtype=np.float32)
    ys_m = np.arange(mw_n, dtype=np.float32)
    xg_m, yg_m = np.meshgrid(xs_m, ys_m)
    bx_m = 0.48*mw_n + (xg_m - mw_n*0.5)*0.15
    band_m = np.exp(-0.5*((yg_m - bx_m)/(mw_n*0.18))**2).astype(np.float32)
    fbm = np.zeros((mw_n, mw_n), dtype=np.float32)
    famp = 1.0
    for fsig in [max(2,mw_n//3), max(2,mw_n//6), max(2,mw_n//12), 2]:
        n_ = rng_mw.standard_normal((mw_n, mw_n)).astype(np.float32)
        n_ = gaussian_filter(n_, sigma=fsig)
        n_ -= n_.min(); n_ /= n_.max()-n_.min()+1e-8
        fbm += famp*n_; famp *= 0.50
    fbm -= fbm.min(); fbm /= fbm.max()+1e-8
    mw_lo = gaussian_filter((band_m*(0.28+0.72*fbm)).astype(np.float32),
                             sigma=max(2, mw_n//20))
    mw_lo /= mw_lo.max()+1e-8
    mw_pil = Image.fromarray((mw_lo*255).astype(np.uint8)).resize((px,px), Image.BILINEAR)
    mw_full = gaussian_filter(np.array(mw_pil,dtype=np.float32)/255.0,
                               sigma=max(4,int(px*0.012)))
    mw_full /= mw_full.max()+1e-8
    # Additive: brightens sky in band region, blue-violet tint
    mw_add = 0.022
    img[:,:,0] += mw_full * mw_add * 0.42
    img[:,:,1] += mw_full * mw_add * 0.55
    img[:,:,2] += mw_full * mw_add * 1.00

    # ── Stars — count from P['bg_n'], seed=42 locked (absolute fixed positions every render) ──
    rng_s = np.random.default_rng(42)
    n_s = P['bg_n']
    sx = rng_s.uniform(0, px, n_s).astype(np.float32)
    sy = rng_s.uniform(0, px, n_s).astype(np.float32)

    # Milky Way band density bias
    band_at = np.exp(-0.5*((sy-(0.48*px+(sx-px*0.5)*0.15))/(px*0.17))**2)

    # Power(0.10): original Moffat distribution — sharp crisp stars
    mag = rng_s.power(0.10, n_s).astype(np.float32)
    mag = np.clip(mag * (1.0 + 0.38*band_at), 0, 1.1).astype(np.float32)
    # NO twinkling — all star positions and brightnesses locked per seed.
    # (frame_idx kept in signature for API compatibility but unused here.)

    # Spectral colors: O/B 2% | A 5% | F/G 15% | K 20% | M 58%
    rv = rng_s.random(n_s).astype(np.float32)
    star_r = np.select([rv<0.02,rv<0.07,rv<0.22,rv<0.42],[0.72,0.87,0.95,0.97],0.93).astype(np.float32)
    star_g = np.select([rv<0.02,rv<0.07,rv<0.22,rv<0.42],[0.76,0.90,0.95,0.93],0.91).astype(np.float32)
    star_b = np.select([rv<0.02,rv<0.07,rv<0.22,rv<0.42],[1.00,0.99,0.97,0.88],0.88).astype(np.float32)

    xi = np.clip(sx.astype(int), 0, px-1)
    yi = np.clip(sy.astype(int), 0, px-1)

    # Sort dim→bright so bright stars overdraw dim ones correctly
    order = np.argsort(mag)

    # ── Atmospheric scintillation — sparse, position-locked ──────────────
    # Real astrophysics: ~6 noticeable twinkle events/sec across whole field.
    # Each event: smooth sine-wave brightness pulse over ~15 frames (~0.6s).
    # Only brightness changes — star positions are absolutely fixed.
    _TW_RATE   = 6.0   # twinkle events/sec visible across entire field
    _TW_FRAMES = 15    # frames per event (~0.6s at 24fps)
    _TW_AMP    = 0.35  # peak brightness variation (±35%)
    _TW_FPS    = 24    # assumed fps for rate calculation
    _tw_period = max(1, int(n_s / _TW_RATE * _TW_FPS))
    _rng_tw = np.random.default_rng(54321)          # separate fixed seed
    _tw_start = _rng_tw.integers(0, _tw_period, n_s)  # each star's event offset
    _tw_raw   = (int(frame_idx) - _tw_start.astype(int)) % _tw_period
    _tw_on    = _tw_raw < _TW_FRAMES
    _tw_phase = np.where(_tw_on, _tw_raw / float(_TW_FRAMES), 0.0).astype(np.float32)
    _tw_mult  = np.where(_tw_on,
                         (1.0 + _TW_AMP * np.sin(np.pi * _tw_phase)),
                         1.0).astype(np.float32)

    for idx in order:
        m = float(mag[idx]) * float(_tw_mult[idx])
        if m < 0.004:
            continue   # essentially invisible — skip

        # Select Moffat stamp for this brightness tier
        if   m < 0.007: lo,hi,stamp = _STAMP_BINS[0]
        elif m < 0.035: lo,hi,stamp = _STAMP_BINS[1]
        elif m < 0.140: lo,hi,stamp = _STAMP_BINS[2]
        elif m < 0.380: lo,hi,stamp = _STAMP_BINS[3]
        elif m < 0.750: lo,hi,stamp = _STAMP_BINS[4]
        else:           lo,hi,stamp = _STAMP_BINS[5]

        hw = stamp.shape[0] // 2
        cx_, cy_ = int(xi[idx]), int(yi[idx])

        # Clipped image bounds
        iy0 = max(0, cy_-hw); iy1 = min(px, cy_+hw+1)
        ix0 = max(0, cx_-hw); ix1 = min(px, cx_+hw+1)
        # Corresponding stamp bounds
        sy0 = iy0-(cy_-hw); sy1 = sy0+(iy1-iy0)
        sx0 = ix0-(cx_-hw); sx1 = sx0+(ix1-ix0)

        s = stamp[sy0:sy1, sx0:sx1] * m
        img[iy0:iy1, ix0:ix1, 0] += s * float(star_r[idx])
        img[iy0:iy1, ix0:ix1, 1] += s * float(star_g[idx])
        img[iy0:iy1, ix0:ix1, 2] += s * float(star_b[idx])

    # ── Sky phenomena ─────────────────────────────────────────────────────
    if sky_events:
        render_sky_events_to_img(img, px, frame_idx, sky_events)
    np.clip(img, 0, 1, out=img)
    ax.imshow(img, extent=[0,W,0,H], origin='lower',
              aspect='auto', interpolation='bilinear', zorder=1.5)

# ══════════════════════════════════════════════════════════════════
# PIXEL-BUFFER STAR RENDERER — shared by soul stars & constellation stars
# Same Moffat PSF approach as the background starfield.
# ══════════════════════════════════════════════════════════════════

def _star_to_buf(buf, cx_w, cy_w, W, H, r_val, g_val, b_val, m_norm,
                 do_spikes=True, do_bloom=True):
    """
    Render a single star into RGBA pixel buffer `buf` using:
      • Moffat PSF core stamp (same kernels as background starfield)
      • Soft Gaussian color bloom (visible halo for bright stars)
      • 6-point diffraction spikes (astrophotography style)

    buf     : (px, px, 4) float32 RGBA — transparent background
    cx_w/cy_w : world-coordinate position (0..W, 0..H)
    r/g/b   : spectral RGB 0..1
    m_norm  : brightness 0..1.5  (0.5=faint soul star, 1.0=bright, 1.3=very bright)
    """
    px = buf.shape[0]
    cx_ = int(cx_w / W * px)
    cy_ = int(cy_w / H * px)
    if not (0 <= cx_ < px and 0 <= cy_ < px):
        return

    # ── Moffat PSF core ───────────────────────────────────────────
    if   m_norm < 0.007: stamp = _STAMP_BINS[0][2]
    elif m_norm < 0.035: stamp = _STAMP_BINS[1][2]
    elif m_norm < 0.140: stamp = _STAMP_BINS[2][2]
    elif m_norm < 0.380: stamp = _STAMP_BINS[3][2]
    elif m_norm < 0.750: stamp = _STAMP_BINS[4][2]
    else:                stamp = _STAMP_BINS[5][2]
    hw = stamp.shape[0] // 2
    iy0 = max(0, cy_-hw); iy1 = min(px, cy_+hw+1)
    ix0 = max(0, cx_-hw); ix1 = min(px, cx_+hw+1)
    sy0 = iy0-(cy_-hw); sy1 = sy0+(iy1-iy0)
    sx0 = ix0-(cx_-hw); sx1 = sx0+(ix1-ix0)
    s = stamp[sy0:sy1, sx0:sx1] * m_norm
    buf[iy0:iy1, ix0:ix1, 0] += s * r_val
    buf[iy0:iy1, ix0:ix1, 1] += s * g_val
    buf[iy0:iy1, ix0:ix1, 2] += s * b_val
    buf[iy0:iy1, ix0:ix1, 3] = np.maximum(buf[iy0:iy1, ix0:ix1, 3], s)

    # ── Soft color bloom (spectral halo) ──────────────────────────
    if do_bloom and m_norm >= 0.30:
        bloom_r  = min(px // 5, max(6, int(m_norm * 55)))
        bloom_sig = float(m_norm * 18.0)
        cy_s = max(0, cy_ - bloom_r); cy_e = min(px, cy_ + bloom_r + 1)
        cx_s = max(0, cx_ - bloom_r); cx_e = min(px, cx_ + bloom_r + 1)
        yy, xx = np.ogrid[cy_s:cy_e, cx_s:cx_e]
        dist2  = ((xx - cx_)**2 + (yy - cy_)**2).astype(np.float32)
        bloom  = (m_norm * 0.38 * np.exp(-dist2 / (2.0 * bloom_sig**2))).astype(np.float32)
        buf[cy_s:cy_e, cx_s:cx_e, 0] += bloom * r_val * 0.65
        buf[cy_s:cy_e, cx_s:cx_e, 1] += bloom * g_val * 0.65
        buf[cy_s:cy_e, cx_s:cx_e, 2] += bloom * b_val * 0.65
        buf[cy_s:cy_e, cx_s:cx_e, 3]  = np.maximum(
            buf[cy_s:cy_e, cx_s:cx_e, 3], bloom * 0.82)

    # ── 6-point diffraction spikes ────────────────────────────────
    if do_spikes and m_norm >= 0.28:
        slen  = min(320, max(10, int(m_norm * 200)))
        d_arr = np.arange(3, slen + 1, dtype=np.float32)
        ivals = (m_norm * 0.18) * np.exp(-d_arr * 3.8 / float(slen))
        for adeg in (0, 30, 60, 90, 120, 150):
            rad_a = math.radians(adeg)
            cdx = math.cos(rad_a); cdy = math.sin(rad_a)
            for sign in (1, -1):
                xsp = np.clip((cx_ + sign*cdx*d_arr + 0.5).astype(np.int32), 0, px-1)
                ysp = np.clip((cy_ + sign*cdy*d_arr + 0.5).astype(np.int32), 0, px-1)
                buf[ysp, xsp, 0] += ivals * r_val
                buf[ysp, xsp, 1] += ivals * g_val
                buf[ysp, xsp, 2] += ivals * b_val
                buf[ysp, xsp, 3]  = np.maximum(buf[ysp, xsp, 3], ivals * 0.80)


# ══════════════════════════════════════════════════════════════════
# POSITION PRE-COMPUTATION
# ══════════════════════════════════════════════════════════════════

def precompute_positions(universe, W, H):
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

def _find_empty_pos(eco_occ, static_occ, excl_r, attempts=800, W=26.0, H=26.0, rng=None, min_eco_gap=1.5):
    """
    Place new element with guaranteed separation from other live ecology elements.
    eco_occ   : [(x,y,r)] currently-alive ecology — hard: gap >= min_eco_gap required
    static_occ: [(x,y,r)] domains + fixed stars  — soft: prefer to avoid, never blocks
    """
    margin = excl_r + 0.8
    best_valid, bv_score = None, -1e9
    best_any,   ba_score = None, -1e9

    for _ in range(attempts):
        x = float(rng.uniform(margin, W - margin))
        y = float(rng.uniform(margin, H - margin))

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
        if eco_gap >= min_eco_gap and score > bv_score:
            bv_score, best_valid = score, (x, y)

    return best_valid or best_any or (W/2, H/2)


def compute_canvas_size(ecology_elements, domains, W, H, min_eco_gap):
    """Print recommended canvas size based on total soul element history."""
    max_concurrent = max(
        sum(1 for el in ecology_elements if el['birth'] <= p < el['death'])
        for p in range(0, 101)
    )
    avg_r = sum(el['excl_r'] for el in ecology_elements) / max(1, len(ecology_elements))
    eco_area = math.pi * (avg_r + min_eco_gap * 0.5)**2 * max_concurrent
    domain_area = sum(math.pi * max(d['rx'], d['ry'])**2 * 0.7 for d in domains)
    total = (domain_area + eco_area) / 0.60
    recommended = max(W, math.sqrt(total))
    print(f"  Soul canvas: {W:.0f}×{H:.0f}  "
          f"(ideal: {recommended:.1f}×{recommended:.1f}, "
          f"peak concurrent ecology: {max_concurrent})")


def precompute_ecology_positions(universe, n_frames, ecology_elements, fixed_pos, W, H, rng, min_eco_gap):
    """
    Simulate ecology frame by frame.  Returns dict: name → (cx, cy)
    Domains are SOFT blockers (0.65× radius) — ecology can appear near their edges,
    filling the whole canvas.  Ecology-ecology gap is HARD-enforced (min_eco_gap).
    """
    positions = {}
    compute_canvas_size(ecology_elements, universe['domains'], W, H, min_eco_gap)

    # Static soft blockers: domain cores + fixed surface stars + constellation centres
    domain_occ = [
        (d['cx'], d['cy'], max(d['rx'], d['ry']) * 0.65)
        for d in universe['domains']
    ]
    star_occ = [
        (sx, sy, 2.8)
        for name, (sx, sy) in fixed_pos.items()
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
            for el in ecology_elements
            if el['name'] in positions and el['birth'] <= pct < el['death']
        ]

        for el in ecology_elements:
            if el['name'] in positions:
                continue
            if el['birth'] <= pct and el['birth'] > prev_pct:
                pos = _find_empty_pos(eco_occ, static_occ, el['excl_r'],
                                      W=W, H=H, rng=rng, min_eco_gap=min_eco_gap)
                positions[el['name']] = pos
                eco_occ.append((pos[0], pos[1], el['excl_r']))

    return positions

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

def build_epoch(universe, intensities, ecology_pct, ecology_elements, ecology_positions):
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
        for el in ecology_elements:
            if el['etype'] != 'neb':
                continue
            i = lerp(el['curve'], ecology_pct)
            if i <= 0.5:
                continue
            pos = ecology_positions.get(el['name'])
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
# RENDER  (positions fixed via fixed_pos — no repulsion, no drift)
# ══════════════════════════════════════════════════════════════════

def render_frame(char, fixed_pos, pct, frame_idx, output_path,
                 fig_sz, dpi, ecology_positions,
                 W, H, bg_temp_curve, aurora_curve,
                 ecology_elements, story_beats, lifecycle_label,
                 sky_events=None):
    p = P; px = int(fig_sz*dpi)
    fig,ax = plt.subplots(figsize=(fig_sz,fig_sz),dpi=dpi)
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.set_xlim(0,W); ax.set_ylim(0,H)
    ax.set_aspect('equal'); ax.axis('off')
    plt.subplots_adjust(0,0,1,1)

    # ── Sky atmosphere ───────────────────────────────────────────
    aurora_surge_mult = 1.0
    if sky_events:
        for evt in sky_events:
            if evt['type'] == 'aurora_surge' and evt['start_frame'] <= frame_idx <= evt['end_frame']:
                dur = max(evt['end_frame'] - evt['start_frame'] + 1, 1)
                phase = (frame_idx - evt['start_frame']) / max(dur-1, 1)
                aurora_surge_mult = 1.0 + (evt['params']['peak_mult'] - 1.0) * math.sin(math.pi * phase)
                break
    render_sky_atmosphere(ax, pct, fig_sz, dpi, W, H, bg_temp_curve, aurora_curve, aurora_surge_mult)

    # ── Starfield (pixel-level Gaussian blobs) ───────────────────
    render_starfield_bg(ax, W, H, px, frame_idx, sky_events)

    # ── Legacy stars — traces of what once mattered ───────────────
    draw_legacy_stars(ax, pct, fig_sz, W, ecology_elements, ecology_positions)

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
        draw_proto_fragment(ax,d['cx'],d['cy'],i,d['bk'],fig_sz,z=2.91,W=W)
    for r_def,v in char['proto_frag_r']:
        x,y = fixed_pos.get(r_def['name']+'_proto', (r_def.get('cx',10),r_def.get('cy',10)))
        draw_proto_fragment(ax,x,y,v,r_def['color'],fig_sz,z=2.93,W=W)

    # ── Proto-clouds (4.5–6.0): condensing glow ──────────────────
    for d,i in char['proto_cloud_d']:
        draw_proto_cloud(ax,d['cx'],d['cy'],d['rx'],d['ry'],i,d['bk'],fig_sz,z=2.97,W=W)
    for r_def,v in char['proto_cloud_r']:
        x,y = fixed_pos.get(r_def['name']+'_proto', (10,10))
        frac=(v-4.5)/3.0; r_=P['sf_outer_r']*0.20*frac; alpha=frac*0.62
        draw_glow(ax,x,y,r_*4.0,r_def['color'],alpha*0.28,fig_sz,z=2.98,W=W)
        draw_star_small(ax,x,y,r_,r_def['color'],alpha*0.55,fig_sz,z=2.99,W=W)

    # ── Full nebulae (≥6.0): fractal clouds ──────────────────────
    if char['nebs']:
        # ── Nebula breath ──
        p_neb = p
        if sky_events:
            for evt in sky_events:
                if evt['type'] == 'nebula_breath':
                    dur = max(evt['end_frame'] - evt['start_frame'] + 1, 1)
                    nb_phase = frame_idx / max(dur - 1, 1)
                    breath = 1.0 + evt['params']['amplitude'] * math.sin(math.pi * nb_phase * 0.5)
                    p_neb = dict(p); p_neb['neb_amax'] = min(1.0, p['neb_amax'] * breath)
                    break
        neb=build_fractal_nebs(p_neb,px,char['nebs'],W,H)
        ax.imshow(neb,extent=[0,W,0,H],origin='lower',
                  interpolation='bilinear',zorder=3,aspect='auto')
        for nd in char['nebs']:
            cx_d,cy_d,mk_col,lbl=nd[0],nd[1],nd[6],nd[10]
            ax.text(cx_d,cy_d,lbl,fontsize=11.0,color='#e8f4ff',alpha=0.68,
                   ha='center',va='center',fontstyle='italic',fontweight='light',zorder=3.5,
                   path_effects=[pe.withStroke(linewidth=5.0,foreground=mk_col)])

    # ── Constellations (FIXED cx,cy — no repulsion) ───────────────
    soul_r=p['cn_soul_r']; reg_r=soul_r*0.38
    for const in char['constellations']:
        cx_c,cy_c = const['cx'],const['cy']   # FIXED, never repelled
        csx=const['sx']; csy=const['sy']
        tmpl=TMPL[const['tmpl']]; soul=const.get('soul',{})
        coords={name:(cx_c+(nx-0.5)*csx,cy_c+(ny-0.5)*csy)
                for name,((nx,ny),*_) in tmpl['stars'].items()}
        for sa,sb in tmpl['lines']:
            x1,y1=coords[sa]; x2,y2=coords[sb]
            ax.plot([x1,x2],[y1,y2],'-',color='#a0b8e8',alpha=p['cl_alpha'],
                   lw=1.8,solid_capstyle='round',zorder=4)
        cst_placed=[]
        for name,((nx,ny),mag,spec_col) in tmpl['stars'].items():
            sx_c,sy_c=coords[name]
            is_soul=name in soul
            if is_soul:
                lbl_txt,soul_col=soul[name]
                draw_star_small(ax,sx_c,sy_c,soul_r,soul_col,p['cn_soul_a'],fig_sz,z=5,W=W)
                off=soul_r*3.5+0.10
                tx,ty,ha,va=find_label_pos(sx_c,sy_c,off,7.5,len(lbl_txt),fig_sz,cst_placed,W,H)
                ax.text(tx,ty,lbl_txt,fontsize=7.5,color=soul_col,alpha=0.85,
                       ha=ha,va=va,fontstyle='italic',fontweight='light',zorder=5,
                       path_effects=[pe.withStroke(linewidth=2.5,foreground='#080808')])
            else:
                draw_star_small(ax,sx_c,sy_c,reg_r,spec_col,p['cn_reg_a']*0.80,fig_sz,z=4.8,W=W)
                sr=reg_r*2.8*0.5
                cst_placed.append((sx_c-sr,sy_c-sr,sx_c+sr,sy_c+sr))
        all_x=[v[0] for v in coords.values()]; all_y=[v[1] for v in coords.values()]
        lbl_cx=sum(all_x)/len(all_x); lbl_cy=sum(all_y)/len(all_y)
        tx_n,ty_n,ha_n,va_n=find_label_pos(lbl_cx,lbl_cy,soul_r*2.5+0.15,
                                            9.0,len(const['name_cn']),fig_sz,cst_placed,W,H)
        ax.text(tx_n,ty_n,const['name_cn'],fontsize=9.0,color='#8090c8',alpha=0.65,
               ha=ha_n,va=va_n,fontweight='light',zorder=4.5,
               path_effects=[pe.withStroke(linewidth=2.5,foreground='#000406')])

    # ── Surface stars (FIXED positions from fixed_pos) ────────────
    sf_r=p['sf_outer_r']
    for lbl,neb_idx,angle_deg,mag,col in char['surf']:
        sx_c,sy_c = fixed_pos[lbl]   # ← ALWAYS same position
        sc=np.clip((mag-6.0)/3.5,0.0,1.0); sc_p=sc**1.8
        r=sf_r*(0.18+0.82*sc_p); lsz=p['sf_lsz']*(0.48+0.52*sc_p)
        draw_star_full(ax,sx_c,sy_c,r,col,p['sf_alpha'],fig_sz,
                      glow_mult=p['sf_glow'],ray_mult=p['sf_ray'],z=7,W=W)
        ty=sy_c-r*p['sf_glow']*0.38-0.08
        ax.text(sx_c,ty,lbl,fontsize=lsz,color=col,alpha=0.92,
               ha='center',va='top',zorder=10,fontweight='light',
               path_effects=[pe.withStroke(linewidth=1.8,foreground='#060606')])

    # ── Ecology stars (v28) ──────────────────────────────────────
    for el in ecology_elements:
        if el['etype'] != 'star':
            continue
        i = lerp(el['curve'], pct)
        if i <= 0.5:
            continue
        pos = ecology_positions.get(el['name'])
        if pos is None:
            continue
        ex, ey = pos
        sc = np.clip((i - 6.0)/3.5, 0.0, 1.0); sc_p = sc**1.8
        if i >= REL_THRESH:
            r  = p['sf_outer_r'] * (0.18 + 0.82*sc_p)
            lsz = p['sf_lsz'] * (0.48 + 0.52*sc_p)
            draw_star_full(ax, ex, ey, r, el['color'], p['sf_alpha'], fig_sz,
                           glow_mult=p['sf_glow'], ray_mult=p['sf_ray'], z=7, W=W)
            ty = ey - r*p['sf_glow']*0.38 - 0.08
            ax.text(ex, ty, el['name'], fontsize=lsz, color=el['color'],
                    alpha=0.88, ha='center', va='top', zorder=10,
                    fontweight='light',
                    path_effects=[pe.withStroke(linewidth=1.8,foreground='#060606')])
        elif i >= 4.5:
            frac = (i-4.5)/3.0
            r_ = p['sf_outer_r']*0.20*frac
            draw_glow(ax, ex, ey, r_*4.0, el['color'], frac*0.28, fig_sz, z=2.98, W=W)
            draw_star_small(ax, ex, ey, r_, el['color'], frac*0.55, fig_sz, z=2.99, W=W)
        elif i >= 1.5:
            draw_proto_fragment(ax, ex, ey, i, el['color'], fig_sz, z=2.93, W=W)

    # ── Ecology constellations (v28) ──────────────────────────────
    soul_r = p['cn_soul_r']; reg_r = soul_r*0.60
    for el in ecology_elements:
        if el['etype'] != 'const':
            continue
        depth = lerp(el['curve'], pct)
        if depth < PAT_THRESH:
            continue
        pos = ecology_positions.get(el['name'])
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
                                p['cn_soul_a'], fig_sz, z=5, W=W)
                off = soul_r*3.5+0.10
                tx,ty,ha,va = find_label_pos(sx_e,sy_e,off,7.5,len(lbl_txt),fig_sz,cst_placed,W,H)
                ax.text(tx,ty,lbl_txt,fontsize=7.5,color=soul_col,alpha=0.85,
                        ha=ha,va=va,fontstyle='italic',fontweight='light',zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.5,foreground='#080808')])
            else:
                draw_star_small(ax, sx_e, sy_e, reg_r, spec_col,
                                p['cn_reg_a']*0.80, fig_sz, z=4.8, W=W)
                sr = reg_r*2.8*0.5
                cst_placed.append((sx_e-sr,sy_e-sr,sx_e+sr,sy_e+sr))
        all_x=[v[0] for v in coords.values()]; all_y=[v[1] for v in coords.values()]
        lbl_cx=sum(all_x)/len(all_x); lbl_cy=sum(all_y)/len(all_y)
        tx_n,ty_n,ha_n,va_n = find_label_pos(lbl_cx,lbl_cy,soul_r*2.5+0.15,
                                              9.0,len(el['name_cn']),fig_sz,cst_placed,W,H)
        ax.text(tx_n,ty_n,el['name_cn'],fontsize=9.0,color='#8090c8',alpha=0.65,
                ha=ha_n,va=va_n,fontweight='light',zorder=4.5,
                path_effects=[pe.withStroke(linewidth=2.5,foreground='#000406')])

    # ── Frame title & story beat ───────────────────────────────────
    best = min(story_beats, key=lambda b: abs(b[0]-pct))
    beat_name, beat_desc = best[1], best[2]
    sentence_approx = round(frame_idx * 100)
    ax.text(W/2, 0.92, f"{char['name']}  灵魂星图",
           fontsize=11.5,color='#aabbd0',alpha=0.65,
           ha='center',va='bottom',fontstyle='italic',zorder=11,
           path_effects=[pe.withStroke(linewidth=3.0,foreground='#000000')])
    ax.text(W/2, 0.40,
           f"第{sentence_approx}句话  ·  【{beat_name}】{beat_desc}",
           fontsize=7.5,color='#7888a0',alpha=0.60,
           ha='center',va='bottom',zorder=11,fontweight='light',
           path_effects=[pe.withStroke(linewidth=2.5,foreground='#000000')])

    # ── Lifecycle annotation (optional) ──────────────────────────────
    if lifecycle_label is not None:
        lc_v = lerp(lifecycle_label['curve'], pct)
        thresh_star, thresh_cloud, thresh_frag = lifecycle_label['thresholds']
        tn = lifecycle_label['tier_names']  # (star, cloud, frag, gone)
        lc_tier = (tn[0] if lc_v >= thresh_star else
                   tn[1] if lc_v >= thresh_cloud else
                   tn[2] if lc_v >= thresh_frag else tn[3])
        ax.text(0.4, H-0.5,
               f"{lifecycle_label['name']}  {lc_v:.1f}  [{lc_tier}]",
               fontsize=6.5, color=lifecycle_label['color'], alpha=0.55,
               ha='left', va='top', zorder=11, fontweight='light',
               path_effects=[pe.withStroke(linewidth=2.0, foreground='#000000')])

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
