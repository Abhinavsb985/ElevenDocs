import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText, Sparkles, Shield, Zap } from 'lucide-react';
import { LoginModal } from './LoginModal';

export const PublicLanding: React.FC = () => {
  const [showLoginModal, setShowLoginModal] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4">
      {/* Header */}
      <header className="max-w-4xl mx-auto mb-8">
        <div className="flex items-center justify-center gap-3">
          <div className="h-12 w-12 bg-primary rounded-xl flex items-center justify-center">
            <FileText className="h-7 w-7 text-primary-foreground" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Text Summarizer
          </h1>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-4xl mx-auto text-center space-y-8">
        <div className="space-y-6">
          <h2 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white leading-tight">
            AI-Powered
            <span className="block text-primary">Text Summarization</span>
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto leading-relaxed">
            Transform long articles, documents, and text into concise, 
            easy-to-understand summaries in seconds using advanced AI technology.
          </p>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <Card className="border-0 shadow-lg bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
            <CardHeader className="text-center">
              <div className="h-12 w-12 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Sparkles className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <CardTitle className="text-gray-900 dark:text-white">AI-Powered</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400 text-center">
                Advanced machine learning algorithms that understand context and meaning
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
            <CardHeader className="text-center">
              <div className="h-12 w-12 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Zap className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <CardTitle className="text-gray-900 dark:text-white">Lightning Fast</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400 text-center">
                Get summaries in seconds, not minutes. Perfect for busy professionals
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
            <CardHeader className="text-center">
              <div className="h-12 w-12 bg-purple-100 dark:bg-purple-900/20 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Shield className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <CardTitle className="text-gray-900 dark:text-white">Secure & Private</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400 text-center">
                Your data is protected with enterprise-grade security and privacy
              </p>
            </CardContent>
          </Card>
        </div>

        {/* CTA */}
        <div className="mt-16">
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-6">
            Ready to get started? Sign in to access the summarization tool.
          </p>
          <div className="flex justify-center">
            <Button 
              size="lg" 
              className="px-8 py-4 text-lg bg-primary hover:bg-primary/90 rounded-xl shadow-lg"
              onClick={() => setShowLoginModal(true)}
            >
              <FileText className="h-5 w-5 mr-2" />
              Start Summarizing
            </Button>
          </div>
        </div>
      </main>

      {/* Login Modal */}
      {showLoginModal && (
        <LoginModal 
          isOpen={showLoginModal}
          onClose={() => setShowLoginModal(false)}
          onSuccess={() => setShowLoginModal(false)}
        />
      )}
    </div>
  );
};
