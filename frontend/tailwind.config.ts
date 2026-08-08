import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        gray: {
          950: "#080d13",
          900: "#0d141d",
          850: "#111923",
          800: "#17212d",
          700: "#2a3747",
          600: "#435165",
          500: "#738095",
          400: "#9ba7b8",
          300: "#c3ccd8",
          200: "#dde3eb",
          100: "#f0f3f7",
        },
      },
    },
  },
  plugins: [],
};

export default config;
