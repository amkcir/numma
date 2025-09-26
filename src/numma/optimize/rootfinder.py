import numpy as np
from math import ceil,log2
from scipy import optimize
import math

def sortedroots(
    f,
    lbound: float,
    rbound: float,
    definition: int,
) -> np.ndarray:
    
    def choose_tolerances(lbound, rbound, *, digits=None):
        """
        返回 (rtol, xtol, dedup_tol, maxiter_per_root) 的合理取值。
        - digits: 目标小数位；None 时用基于机器精度的默认。
        """
        L = abs(lbound)
        R = abs(rbound)
        scale_x = max(1.0, L, R)
        eps = np.finfo(float).eps
        if digits is not None:
            rtol = 10.0**(-digits)
            xtol = rtol * scale_x
        else:
            rtol = np.sqrt(eps)            # ~1e-8
            xtol = rtol * scale_x          # 绝对容差

        # 二分迭代次数估计
        width = rbound - lbound
        if width <= 0:
            raise ValueError("rbound must be > lbound")
        N = max(1, ceil(log2(width / max(xtol, np.spacing(width)))))

        # 去重容差（放宽 10 倍，避免重复）
        dedup_tol = 10.0 * max(xtol, rtol * scale_x, np.spacing(scale_x))

        maxiter_per_root = N + 5
        return rtol, xtol, dedup_tol, maxiter_per_root


    def zero_atol_from_samples(fs):
        """根据采样值自适应生成零点判据的 atol。"""
        fs = np.asarray(fs)
        fs = fs[np.isfinite(fs)]
        if fs.size == 0:
            return 0.0
        fscale = np.median(np.abs(fs)) or 1.0
        eps = np.finfo(float).eps
        return max(np.sqrt(eps) * fscale, 10*np.spacing(fscale))

    def deriv_tol_from_samples(fs, xs, *, frac: float = 0.2, seed: int | None = None) -> float:
        """
        根据采样值和步长，自适应生成导数比较容差。
        - 随机抽取 frac (默认 1/5) 的区间来估计典型 |df/dx|。
        """
        fs = np.asarray(fs, dtype=float)
        xs = np.asarray(xs, dtype=float)

        mask_f = np.isfinite(fs)
        idx = np.nonzero(mask_f[1:] & mask_f[:-1])[0] + 1  # 区间右端索引
        nseg = idx.size
        if nseg == 0:
            return 1e-6

        # 随机抽样 1/5 区间（至少 1 个）
        rng = np.random.default_rng(seed)
        k = max(1, int(frac * nseg))
        pick = rng.choice(idx, size=k, replace=False)

        # 差分斜率
        dfi = np.abs(fs[pick] - fs[pick - 1])
        dxi = np.abs(xs[pick] - xs[pick - 1])
        safe_dxi = np.maximum(dxi, np.finfo(float).eps * (1 + np.abs(xs[pick])))
        slopes = abs(dfi / safe_dxi)
        slopes = slopes[np.isfinite(slopes)]

        if slopes.size == 0:
            return 1e-6

        # 典型导数量级（中位数）
        dscale = float(np.median(slopes))
        """
        if dscale < np.finfo(float).eps:  # 保底，避免太小
            dscale = np.finfo(float).eps
        eps = np.finfo(float).eps
        """
        return dscale/100

    rtol, xtol, dedup_tol, maxiter_per_root = choose_tolerances(lbound, rbound, digits=12)
    zero_as_bracket_eps = (rbound-lbound)/(10*definition)
    """
    用等分采样收集变号区间，并用二分法逐一求根（线性收敛、最稳健）。
    要求：f(x) 在定义域外/奇点处返回 np.nan 或抛异常（本函数会忽略这些点/区间）。
    """
    if not np.isfinite(lbound) or not np.isfinite(rbound) or rbound <= lbound:
        raise ValueError("需要有限且满足 rbound > lbound 的边界。")
    if definition < 2:
        raise ValueError("definition 应 ≥ 2。")

    xs = np.linspace(lbound, rbound, definition)

    def safe_eval(x):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            try:
                y = float(f(x))   # 你的原函数 f
            except Exception:
                return np.nan
        return y if np.isfinite(y) else np.nan


    fs = np.array([safe_eval(x) for x in xs])
    finite = np.isfinite(fs)



    # —— 收集括号（变号子区间） ——
    brackets = []

    # A) 采样点恰为零：在该点左右造一个极窄括号
    zero_atol = zero_atol_from_samples(fs)
    zeros_idx = np.where(np.isfinite(fs) & np.isclose(fs, 0.0, atol=zero_atol))[0]
    for i in zeros_idx:
        x0 = xs[i]
        if lbound < x0 < rbound:
            a = max(lbound, x0 - zero_as_bracket_eps)
            b = min(rbound, x0 + zero_as_bracket_eps)
            if a < x0 < b:
                brackets.append((a, b))

    dfs = np.diff(fs) / np.diff(xs)
    deriv_tol = deriv_tol_from_samples(fs, xs)
    # B) 相邻点均有限 → 检测严格变号
    for i in range(1,len(xs) - 2):
        a, b = xs[i], xs[i + 1]
        fa, fb = fs[i], fs[i + 1]
        # 如果任一端不是有限数 → 跳过
        if not (finite[i] and finite[i + 1]):
            continue
        # 如果任一端索引在 zeros_idx 里 → 跳过
        if i in zeros_idx or (i + 1) in zeros_idx:
            continue

        if np.sign(fa) != np.sign(fb):
        # 导数一致性：左端用 dfs[i-1]，右端用 dfs[i]（二者都存在）
            da = dfs[i - 1]
            db = dfs[i]

            if np.isfinite(da) and np.isfinite(db):
                # 相对阈值：|da-db| ≤ tol*(1+max(|da|,|db|))
                if abs(da - db) <= deriv_tol * (1.0 + max(abs(da), abs(db))):
                    # 可选：微收缩避免端点卡奇点
                    brackets.append((a, b))



    # —— 二分法求根 ——
    roots = []

    def add_root(x):
        if not (lbound < x < rbound):
            return
        for r in roots:
            if abs(x - r) <= dedup_tol * (1 + abs(r)):
                return
        roots.append(x)

    rtol_trans=np.float64(rtol)
    for (a, b) in brackets:
        root = optimize.brentq(f, a, b, xtol=xtol, rtol=rtol_trans, maxiter=10000,full_output=False)
        if root is not None and math.isfinite(root):
            add_root(root)

    roots.sort()
    return np.array(roots)