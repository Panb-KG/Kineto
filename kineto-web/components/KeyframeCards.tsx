/**
 * components/KeyframeCards.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 四宫格教学图：展示 4 个关键帧的截图 + 骨骼标注。
 *
 * 图片由引擎 kineto_core.py 的 _generate_grid_images() 生成，
 * 保存到 job 目录并通过 /api/jobs/{id}/grid_XX.jpg 访问。
 */

"use client";

import { API_BASE } from "../lib/api";

interface KeyframeCardsProps {
  jobId: string;
  gridImages?: string[];
  gridLabels?: string[];
}

export default function KeyframeCards({ jobId, gridImages, gridLabels }: KeyframeCardsProps) {
  if (!gridImages || gridImages.length === 0) {
    return null;
  }

  const labels = gridLabels || ["关键帧 1", "关键帧 2", "关键帧 3", "关键帧 4"];

  return (
    <section className="keyframe-cards" aria-label="四宫格教学图">
      <h2 className="section-label">动作分解 · Keyframe Breakdown</h2>
      <div className="keyframe-cards__grid">
        {gridImages.map((img, idx) => (
          <div key={idx} className="keyframe-cards__card">
            <div className="keyframe-cards__number">{String(idx + 1).padStart(2, "0")}</div>
            <img
              src={`${API_BASE}/jobs/${jobId}/${img}`}
              alt={labels[idx] || `关键帧 ${idx + 1}`}
              className="keyframe-cards__image"
              loading="lazy"
            />
            <div className="keyframe-cards__label">{labels[idx]}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
