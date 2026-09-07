/**
 * app/page.tsx —— 首页（服务端组件）。
 *
 * 组合：页头 → 上半部分 4 张关键帧指导图（占位）→ 下半部分 3D 骨架交互视图
 * + 时间轴控制 + 元数据面板。所有数据相关的交互逻辑封装在客户端组件
 * <ViewerStage/> 中，本文件只负责静态排版外壳。
 *
 * 支持通过查询参数 ?job=<id> 指定后端任务；缺省时离线加载内置 fixture。
 */

import ViewerStage from "../components/ViewerStage";
import UploadPanel from "../components/UploadPanel";

export const dynamic = "force-dynamic";

interface PageProps {
  searchParams?: { job?: string };
}

export default function Page({ searchParams }: PageProps) {
  const jobId = searchParams?.job;

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand">
          <span className="brand__mark">
            Kin<em>e</em>to
          </span>
          <span className="brand__sub">Movement Rehab · 3D Pose Lab</span>
        </div>
        <div className="header-meta">
          <span className="pill">
            <span className="pill__dot" aria-hidden />
            云边协同 · Cloudflare Tunnel
          </span>
          <p className="header-meta__note">
            4DHumans 逆向解析真人教学视频为三维姿态，交互式骨架辅助《肌骨解剖触诊》
            中的肩胛节律与骨盆中立位展示。
          </p>
        </div>
      </header>

      <main>
        <UploadPanel />

        <h1 className="section-label">关键帧指导图 · Guidance Frames</h1>
        <ViewerStage jobId={jobId} />
      </main>

      <footer className="site-footer">
        <span>Kineto Engine · 4DHumans + ComfyUI/FLUX · Zeabur Cloud</span>
        <span>SMPL 24-joint skeleton · MVP</span>
      </footer>
    </div>
  );
}
