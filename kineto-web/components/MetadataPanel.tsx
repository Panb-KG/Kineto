/**
 * components/MetadataPanel.tsx
 * ─────────────────────────────────────────────────────────────────────────
 * 元数据面板：展示 fps / total_frames / final_quality_score / extraction_mode
 * 等关键信息。当 extraction_mode != '4dhumans' 时显示醒目的告警徽章
 * （假数据守卫，镜像后端的 fake-data guard 逻辑）。
 */

import type { PoseMetadata } from "../lib/types";
import { JOINT_ORDER_CANONICAL } from "../lib/types";
import type { PoseDataSource } from "../lib/poseData";

interface MetadataPanelProps {
  metadata: PoseMetadata;
  source: PoseDataSource;
  fallbackReason?: string;
}

/** final_quality_score 嵌套于 metadata.pipeline（已核对真实样本）。 */
function qualityScore(m: PoseMetadata): number | undefined {
  return m.pipeline?.final_quality_score;
}

export default function MetadataPanel({
  metadata,
  source,
  fallbackReason,
}: MetadataPanelProps) {
  const isFakeData = metadata.extraction_mode !== "4dhumans";
  const score = qualityScore(metadata);
  // M3: joint_order 非 canonical 时显示告警
  const jointOrderWarn =
    metadata.joint_order !== undefined &&
    metadata.joint_order !== JOINT_ORDER_CANONICAL;

  return (
    <aside className="meta-panel" aria-label="数据元信息">
      <div className="meta-head">
        <h2>数据元信息</h2>
        <span className={`meta-source meta-source--${source}`}>
          {source === "api" ? "LIVE API" : "FIXTURE"}
        </span>
      </div>

      {isFakeData && (
        <div className="meta-warning" role="alert">
          <span className="meta-warning__icon" aria-hidden>
            ⚠
          </span>
          <div>
            <strong>非 4DHumans 数据</strong>
            <p>
              extraction_mode = <code>{String(metadata.extraction_mode)}</code>
              ，可能为占位/模拟数据，仅供界面演示，不可用于临床判断。
            </p>
          </div>
        </div>
      )}

      {/* M3: joint_order 非 canonical 序告警 */}
      {jointOrderWarn && (
        <div className="meta-warning" role="alert">
          <span className="meta-warning__icon" aria-hidden>
            ⚠
          </span>
          <div>
            <strong>非标准关节序产物</strong>
            <p>
              joint_order = <code>{metadata.joint_order}</code>（非{" "}
              <code>{JOINT_ORDER_CANONICAL}</code>
              ），骨架渲染基于 canonical 序镜像，结果可能偏斜。
            </p>
          </div>
        </div>
      )}

      {fallbackReason && (
        <p className="meta-fallback" title={fallbackReason}>
          {fallbackReason}
        </p>
      )}

      <dl className="meta-grid">
        <div className="meta-item">
          <dt>video_fps</dt>
          <dd>{metadata.video_fps?.toFixed(2)}</dd>
        </div>
        <div className="meta-item">
          <dt>total_frames</dt>
          <dd>{metadata.total_frames}</dd>
        </div>
        <div className="meta-item">
          <dt>quality</dt>
          <dd>
            {typeof score === "number" ? score.toFixed(3) : "—"}
            {typeof score === "number" && (
              <span className="meta-bar" aria-hidden>
                <span style={{ width: `${Math.round(score * 100)}%` }} />
              </span>
            )}
          </dd>
        </div>
        <div className="meta-item">
          <dt>extraction_mode</dt>
          <dd className={isFakeData ? "meta-flag" : undefined}>
            {String(metadata.extraction_mode)}
          </dd>
        </div>
        <div className="meta-item">
          <dt>resolution</dt>
          <dd>{metadata.resolution ?? "—"}</dd>
        </div>
        <div className="meta-item">
          <dt>device</dt>
          <dd>{metadata.device ?? "—"}</dd>
        </div>
        <div className="meta-item meta-item--wide">
          <dt>model_version</dt>
          <dd>{metadata.model_version ?? "—"}</dd>
        </div>
        {metadata.joint_order !== undefined && (
          <div className="meta-item">
            <dt>joint_order</dt>
            <dd className={jointOrderWarn ? "meta-flag" : undefined}>
              {metadata.joint_order}
            </dd>
          </div>
        )}
        {metadata.schema_version !== undefined && (
          <div className="meta-item">
            <dt>schema_version</dt>
            <dd>{metadata.schema_version}</dd>
          </div>
        )}
        {metadata.video_md5 && (
          <div className="meta-item meta-item--wide">
            <dt>video_md5</dt>
            <dd style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
              {metadata.video_md5}
            </dd>
          </div>
        )}
      </dl>
    </aside>
  );
}
