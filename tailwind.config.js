/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{njk,md,html}", "./_site/**/*.html"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#f9f9f8",
          dim: "#d4dcda",
          bright: "#f9f9f8",
          "container-lowest": "#ffffff",
          "container-low": "#f2f4f3",
          container: "#ebeeed",
          "container-high": "#e4e9e8",
          "container-highest": "#dde4e3",
          variant: "#dde4e3",
          tint: "#545f73",
        },
        primary: {
          DEFAULT: "#545f73",
          dim: "#485367",
          container: "#d8e3fb",
          fixed: "#d8e3fb",
          "fixed-dim": "#cad5ed",
        },
        secondary: {
          DEFAULT: "#5450c1",
          dim: "#4843b4",
          container: "#e2dfff",
          fixed: "#e2dfff",
          "fixed-dim": "#d3d0ff",
        },
        tertiary: {
          DEFAULT: "#5d5d78",
          dim: "#51516c",
          container: "#d9d7f8",
          fixed: "#d9d7f8",
          "fixed-dim": "#cbc9e9",
        },
        on: {
          surface: "#2d3433",
          "surface-variant": "#5a6060",
          primary: "#f6f7ff",
          "primary-container": "#475266",
          secondary: "#fbf7ff",
          "secondary-container": "#4741b3",
          background: "#2d3433",
          error: "#fff7f6",
        },
        outline: {
          DEFAULT: "#757c7b",
          variant: "#adb3b2",
        },
        background: "#f9f9f8",
        error: "#9f403d",
      },
      fontFamily: {
        headline: ["Newsreader", "serif"],
        body: ["Inter", "sans-serif"],
        label: ["Inter", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg: "0.25rem",
        xl: "0.5rem",
        full: "0.75rem",
      },
    },
  },
  plugins: [],
};
