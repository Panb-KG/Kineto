/**
 * app/layout.tsx —— 根布局：字体装配与全局元信息。
 *
 * 字体搭配（拒绝 Inter/Roboto 等通用脸）：
 *   - 标题 / 品牌：Instrument Serif（编辑级衬线，赋予临床报告以排版气质）
 *   - 正文：IBM Plex Sans（清晰、专业、技术感）
 *   - 数据 / 读数：IBM Plex Mono（等宽，数值对齐，工程仪表感）
 *
 * 采用运行时 <link> 加载 Google Fonts 而非 next/font/google：
 * 构建期不产生网络依赖（离线可构建），联网时自动下载，离线时优雅回退到
 * 系统字体栈（见 globals.css 的 --font-* 变量）。
 */

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kineto · 运动康复 3D 姿态解析",
  description:
    "基于 4DHumans 的三维人体姿态可视化平台，以医疗极简交互呈现康复动作关键帧与可拖拽 3D 骨架。",
};

const GOOGLE_FONTS_HREF =
  "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link rel="stylesheet" href={GOOGLE_FONTS_HREF} />
      </head>
      <body>{children}</body>
    </html>
  );
}
