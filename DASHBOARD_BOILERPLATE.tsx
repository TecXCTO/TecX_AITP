"""
AI Trading Platform - Web Dashboard
React + shadcn/ui + Recharts

Mobile-responsive real-time trading dashboard
"""

// ============================================
// package.json
// ============================================
{
  "name": "ai-trading-dashboard",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "@tanstack/react-query": "^5.17.0",
    "axios": "^1.6.5",
    "recharts": "^2.10.0",
    "lucide-react": "^0.309.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "class-variance-authority": "^0.7.0",
    "@radix-ui/react-alert-dialog": "^1.0.5",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-label": "^2.0.2",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-slider": "^1.1.2",
    "@radix-ui/react-switch": "^1.0.3",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
    "socket.io-client": "^4.6.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.48",
    "@types/react-dom": "^18.2.18",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "postcss": "^8.4.33",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.11"
  }
}

// ============================================
// src/App.tsx - Main Application Component
// ============================================
import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Activity, DollarSign, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
interface PortfolioData {
  total_value: number;
  cash: number;
  positions_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_return: number;
}

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

interface BotStatus {
  bot_id: string;
  name: string;
  status: 'running' | 'stopped' | 'paused';
  strategy: string;
  trades_today: number;
  pnl_today: number;
}

interface RiskParams {
  stop_loss_pct: number;
  max_drawdown_pct: number;
  max_position_size: number;
}

function App() {
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [riskParams, setRiskParams] = useState<RiskParams>({
    stop_loss_pct: 2.0,
    max_drawdown_pct: 10.0,
    max_position_size: 10000,
  });

  const queryClient = useQueryClient();

  // Fetch portfolio data
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio'],
    queryFn: async () => {
      const { data } = await api.get<PortfolioData>('/api/portfolio');
      return data;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch positions
  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      const { data } = await api.get<Position[]>('/api/positions');
      return data;
    },
    refetchInterval: 5000,
  });

  // Fetch bots
  const { data: bots } = useQuery({
    queryKey: ['bots'],
    queryFn: async () => {
      const { data } = await api.get<BotStatus[]>('/api/bots');
      return data;
    },
    refetchInterval: 3000,
  });

  // Toggle bot mutation
  const toggleBot = useMutation({
    mutationFn: async (botId: string) => {
      const bot = bots?.find(b => b.bot_id === botId);
      const newStatus = bot?.status === 'running' ? 'stopped' : 'running';
      await api.post(`/api/bots/${botId}/toggle`, { status: newStatus });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bots'] });
    },
  });

  // Update risk params mutation
  const updateRiskParams = useMutation({
    mutationFn: async (params: RiskParams) => {
      await api.post('/api/risk/update', params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk'] });
    },
  });

  // Mock equity curve data
  const equityCurveData = Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    value: 100000 + (Math.random() - 0.5) * 5000 + i * 100,
  }));

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              AI Trading Platform
            </h1>
            <div className="flex items-center gap-4">
              <Button variant="outline">
                <Activity className="mr-2 h-4 w-4" />
                Live
              </Button>
              <Button variant="destructive">Emergency Stop</Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Portfolio Summary */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Value</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${portfolio?.total_value.toLocaleString() || '0'}
              </div>
              <p className="text-xs text-muted-foreground">
                Cash: ${portfolio?.cash.toLocaleString() || '0'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Daily P/L</CardTitle>
              {(portfolio?.daily_pnl || 0) >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-600" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-600" />
              )}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${(portfolio?.daily_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${portfolio?.daily_pnl.toFixed(2) || '0.00'}
              </div>
              <p className="text-xs text-muted-foreground">
                {portfolio?.daily_pnl_pct.toFixed(2)}% today
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Open Positions</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {positions?.length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Value: ${portfolio?.positions_value.toLocaleString() || '0'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Return</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {portfolio?.total_return.toFixed(2)}%
              </div>
              <p className="text-xs text-muted-foreground">
                All-time
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Risk Alert */}
        {(portfolio?.daily_pnl || 0) < -500 && (
          <Alert variant="destructive" className="mb-8">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Risk Warning</AlertTitle>
            <AlertDescription>
              Daily loss exceeds $500. Consider reducing exposure or pausing trading.
            </AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="positions">Positions</TabsTrigger>
            <TabsTrigger value="bots">Trading Bots</TabsTrigger>
            <TabsTrigger value="risk">Risk Settings</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Equity Curve (24h)</CardTitle>
                <CardDescription>Portfolio value over the last 24 hours</CardDescription>
              </CardHeader>
              <CardContent className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equityCurveData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={['auto', 'auto']} />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Positions Tab */}
          <TabsContent value="positions" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Open Positions</CardTitle>
                <CardDescription>Your current trading positions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {positions?.map((position) => (
                    <div key={position.symbol} className="flex items-center justify-between p-4 border rounded-lg">
                      <div>
                        <p className="font-semibold">{position.symbol}</p>
                        <p className="text-sm text-muted-foreground">
                          {position.quantity} @ ${position.entry_price.toFixed(2)}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className={`font-semibold ${position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          ${position.unrealized_pnl.toFixed(2)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {position.unrealized_pnl_pct.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  ))}
                  {!positions?.length && (
                    <p className="text-center text-muted-foreground py-8">
                      No open positions
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Bots Tab */}
          <TabsContent value="bots" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Trading Bots</CardTitle>
                <CardDescription>Manage your automated trading strategies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {bots?.map((bot) => (
                    <div key={bot.bot_id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex-1">
                        <p className="font-semibold">{bot.name}</p>
                        <p className="text-sm text-muted-foreground">{bot.strategy}</p>
                        <div className="flex gap-4 mt-2 text-xs">
                          <span>Trades: {bot.trades_today}</span>
                          <span className={bot.pnl_today >= 0 ? 'text-green-600' : 'text-red-600'}>
                            P/L: ${bot.pnl_today.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className={`px-2 py-1 rounded text-xs ${
                          bot.status === 'running' ? 'bg-green-100 text-green-800' :
                          bot.status === 'stopped' ? 'bg-red-100 text-red-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {bot.status}
                        </span>
                        <Switch
                          checked={bot.status === 'running'}
                          onCheckedChange={() => toggleBot.mutate(bot.bot_id)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Risk Settings Tab */}
          <TabsContent value="risk" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Risk Parameters</CardTitle>
                <CardDescription>Adjust your risk management settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Stop Loss (%)</label>
                  <Slider
                    value={[riskParams.stop_loss_pct]}
                    onValueChange={([value]) => setRiskParams({ ...riskParams, stop_loss_pct: value })}
                    max={10}
                    step={0.5}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Current: {riskParams.stop_loss_pct.toFixed(1)}%
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Drawdown (%)</label>
                  <Slider
                    value={[riskParams.max_drawdown_pct]}
                    onValueChange={([value]) => setRiskParams({ ...riskParams, max_drawdown_pct: value })}
                    max={20}
                    step={1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Current: {riskParams.max_drawdown_pct.toFixed(1)}%
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Position Size ($)</label>
                  <Slider
                    value={[riskParams.max_position_size]}
                    onValueChange={([value]) => setRiskParams({ ...riskParams, max_position_size: value })}
                    min={1000}
                    max={50000}
                    step={1000}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Current: ${riskParams.max_position_size.toLocaleString()}
                  </p>
                </div>

                <Button onClick={() => updateRiskParams.mutate(riskParams)} className="w-full">
                  Update Risk Settings
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;

// ============================================
// tailwind.config.js
// ============================================
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
