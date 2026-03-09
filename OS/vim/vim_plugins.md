C++ / CUDA / embedded / remote / Python / CMake
假設：Neovim ≥ 0.9、Linux、SSH 遠端開發。

1. 目錄結構

``` text
~/.config/nvim/
├── init.lua
├── lua/
│   ├── plugins/
│   │   ├── core.lua
│   │   ├── lsp.lua
│   │   ├── cmp.lua
│   │   ├── treesitter.lua
│   │   ├── cmake.lua
│   │   ├── git.lua
│   │   ├── ui.lua
│   │   └── tools.lua
│   └── config/
│       └── lsp.lua
```

2. init.lua

``` lua
-- ~/.config/nvim/init.lua

-- leader key
vim.g.mapleader = " "
vim.g.maplocalleader = " "

-- 基本設定
vim.opt.number = true
vim.opt.relativenumber = true
vim.opt.termguicolors = true
vim.opt.mouse = "a"
vim.opt.clipboard = "unnamedplus"

-- lazy.nvim bootstrap
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable", lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup("plugins")
```

3. plugins

``` lua
-- ~/.config/nvim/lua/plugins/core.lua
return {
  { "nvim-lua/plenary.nvim" },

  {
    "nvim-telescope/telescope.nvim",
    dependencies = { "plenary.nvim" },
    cmd = "Telescope",
  },

  {
    "folke/which-key.nvim",
    event = "VeryLazy",
  },
}
```

4. LSP

``` lua
-- ~/.config/nvim/lua/plugins/lsp.lua
return {
  {
    "williamboman/mason.nvim",
    build = ":MasonUpdate",
  },
  {
    "williamboman/mason-lspconfig.nvim",
  },
  {
    "neovim/nvim-lspconfig",
    config = function()
      require("config.lsp")
    end,
  },
}
```

LSP 設定

``` lua
-- ~/.config/nvim/lua/config/lsp.lua
local lspconfig = require("lspconfig")

require("mason").setup()
require("mason-lspconfig").setup({
  ensure_installed = {
    "clangd",
    "pyright",
  },
})

-- clangd：C / C++ / CUDA
lspconfig.clangd.setup({
  cmd = {
    "clangd",
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
  },
})

-- python
lspconfig.pyright.setup({})
```

⚠️ 重點現實提醒
C++ / CUDA 沒有 compile_commands.json → LSP 就是裝飾。

``` cmake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
```

5. 補完系統

``` lua
-- ~/.config/nvim/lua/plugins/cmp.lua
return {
  {
    "hrsh7th/nvim-cmp",
    event = "InsertEnter",
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-buffer",
      "hrsh7th/cmp-path",
    },
    config = function()
      local cmp = require("cmp")
      cmp.setup({
        mapping = cmp.mapping.preset.insert({
          ["<CR>"] = cmp.mapping.confirm({ select = true }),
        }),
        sources = {
          { name = "nvim_lsp" },
          { name = "buffer" },
          { name = "path" },
        },
      })
    end,
  },
}
```

6.Treesitter 

``` lua
-- ~/.config/nvim/lua/plugins/treesitter.lua
return {
  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate",
    config = function()
      require("nvim-treesitter.configs").setup({
        ensure_installed = {
          "c", "cpp", "cuda",
          "python", "cmake",
          "bash", "json", "yaml",
        },
        highlight = { enable = true },
      })
    end,
  },
}
```

7. cmake

``` lua
-- ~/.config/nvim/lua/plugins/cmake.lua
return {
  {
    "Civitasv/cmake-tools.nvim",
    ft = { "c", "cpp", "cmake" },
    config = function()
      require("cmake-tools").setup({
        cmake_build_directory = "build/${variant:buildType}",
      })
    end,
  },
}
```

8. Git

``` lua
-- ~/.config/nvim/lua/plugins/git.lua
return {
  { "tpope/vim-fugitive", cmd = { "Git" } },
  {
    "lewis6991/gitsigns.nvim",
    event = "BufRead",
    config = true,
  },
}
```

9.工具與終端

``` lua
-- ~/.config/nvim/lua/plugins/tools.lua
return {
  {
    "akinsho/toggleterm.nvim",
    version = "*",
    config = true,
  },
  { "numToStr/Comment.nvim", config = true },
  { "kylechui/nvim-surround", config = true },
}
```

這是 embedded / remote debug 的主戰場。

10. Ui

``` lua
-- ~/.config/nvim/lua/plugins/ui.lua
return {
  {
    "folke/tokyonight.nvim",
    priority = 1000,
    config = function()
      vim.cmd.colorscheme("tokyonight")
    end,
  },
}
```

11. 十一、安裝流程（照做）
# 1. 安裝 neovim（建議官方或 AppImage）

``` bash
nvim --version
```

# 2. 建立設定目錄

``` bash
mkdir -p ~/.config/nvim/lua/{plugins,config}
```

# 3. 貼上以上檔案
# 4. 啟動 nvim

``` bash
nvim
```

第一次啟動會自動：

- clone lazy.nvim

- 安裝所有插件

- 裝 Mason LSP

檢查：

``` vim
:LspInfo
:Mason
```