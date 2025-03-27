import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/DocsWebsite',
  head: [['link', { rel: 'icon', href: '/cat_fish.svg' }]],
  title: "BoBolilla's site",
  description: "BoBolilla的笔记记录网站",
  cleanUrls: true, //从 URL 中删除 .html 后缀
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: {
      light: '/logo.svg',
      dark: '/logo.svg',
      alt: '网站 Logo'
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: '笔记', link: '/笔记/' }
    ],

    sidebar: [
      {
        text: '目录',
        items: [
          { text: 'Happy Birthday!', link: '/笔记/' },
          { text: 'English', link: '/笔记/英语' }
          
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/BoBolilla' }
    ]
  }
})
