/**
 * components/KeyframeCards.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 页面上半部分：4 张静态关键帧动作指导图。
 *
 * ⚠️ 占位实现：真实的 2D 标准图由后端 ComfyUI / FLUX.1 管线生成（PROJECT_PLAN
 * 任务 1.2），不在本期前端范围内。此处使用纯 CSS/SVG 占位卡，保证完全离线可用，
 * 待后端就绪后把 src 换成返回的图片 URL 即可。
 */

interface KeyframeCardProps {
  index: number;
  label: string;
  cue: string;
}

const CARDS: KeyframeCardProps[] = [
  { index: 1, label: "起始位 · Setup", cue: "骨盆中立，肩胛下沉" },
  { index: 2, label: "离心 · Descent", cue: "髋膝同步屈曲，躯干稳定" },
  { index: 3, label: "转折点 · Amortization", cue: "触底瞬间，膝不过度内扣" },
  { index: 4, label: "向心 · Ascent", cue: "伸髋发力，回到中立位" },
];

function PlaceholderArt({ index }: { index: number }) {
  // 极简线稿「人形」占位图，随序号改变姿态角度，暗示动作阶段
  const angles = [-8, 14, 30, 10];
  const a = angles[(index - 1) % angles.length];
  return (
    <svg
      className="kf-art"
      viewBox="0 0 120 120"
      role="img"
      aria-label={`关键帧 ${index} 占位示意图`}
    >
      <defs>
        <linearGradient id={`kfg-${index}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#eef3f8" />
          <stop offset="100%" stopColor="#e2eaf2" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="120" height="120" fill={`url(#kfg-${index})`} />
      <g
        stroke="#8fa6bb"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        transform={`rotate(${a} 60 60)`}
      >
        <circle cx="60" cy="30" r="8" />
        <line x1="60" y1="38" x2="60" y2="70" />
        <line x1="60" y1="46" x2="44" y2="62" />
        <line x1="60" y1="46" x2="76" y2="62" />
        <line x1="60" y1="70" x2="50" y2="94" />
        <line x1="60" y1="70" x2="70" y2="94" />
      </g>
      <text
        x="60"
        y="112"
        textAnchor="middle"
        fontSize="8"
        fill="#9fb0c0"
        fontFamily="monospace"
      >
        COMFYUI / FLUX · PENDING
      </text>
    </svg>
  );
}

export default function KeyframeCards() {
  return (
    <section className="kf-grid" aria-label="关键帧动作指导图">
      {CARDS.map((card) => (
        <figure className="kf-card" key={card.index}>
          <div className="kf-thumb">
            <PlaceholderArt index={card.index} />
            <span className="kf-badge">{card.index.toString().padStart(2, "0")}</span>
          </div>
          <figcaption className="kf-caption">
            <span className="kf-label">{card.label}</span>
            <span className="kf-cue">{card.cue}</span>
          </figcaption>
        </figure>
      ))}
    </section>
  );
}
