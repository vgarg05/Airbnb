/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        airbnb: {
          50:  '#fff1f2',
          100: '#ffe0e3',
          200: '#ffc6cc',
          300: '#ff9aa4',
          400: '#ff6375',
          500: '#FF385C',
          600: '#ed1a3d',
          700: '#c80f2f',
          800: '#a6102c',
          900: '#8a1229',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in':    'fadeIn 0.3s ease-out',
        'slide-up':   'slideUp 0.35s cubic-bezier(0.16,1,0.3,1)',
        'bounce-dot': 'bounceDot 1.2s infinite ease-in-out',
        'shimmer':    'shimmer 1.8s infinite linear',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        bounceDot: {
          '0%, 80%, 100%': { transform: 'scale(0)', opacity: '0.4' },
          '40%':            { transform: 'scale(1)',   opacity: '1'   },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition:  '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
