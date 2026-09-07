/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  // R3F / three 均为客户端 ESM 依赖，交给 Next 编译管线处理，避免 ESM 互操作问题
  transpilePackages: ["three"],
  eslint: {
    // 构建阶段不因 lint 规则中断（本项目未接入 eslint 配置）
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig;
