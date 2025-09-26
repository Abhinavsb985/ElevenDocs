import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useApi } from '@/hooks/useApi';
import { useAuth } from '@/contexts/AuthContext';

export const BackendTest: React.FC = () => {
  const [text, setText] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { post } = useApi();
  const { isAuthenticated } = useAuth();

  const testHealth = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch('http://127.0.0.1:8002/');
      const data = await response.json();
      
      if (response.ok) {
        setSummary(`Health check: ${JSON.stringify(data)}`);
      } else {
        setError(`Health check failed: ${data.error}`);
      }
    } catch (err) {
      setError(`Health check error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const testSummarize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to summarize');
      return;
    }

    try {
      setLoading(true);
      setError('');
      
      const data = await post('/summarize', { text });
      setSummary(data.summary);
    } catch (err) {
      setError(`Summarize error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto space-y-4">
      <h3 className="text-lg font-semibold">Backend Connection Test</h3>
      
      <div className="space-y-2">
        <Button 
          onClick={testHealth} 
          disabled={loading}
          variant="outline"
          className="w-full"
        >
          Test Health Endpoint
        </Button>
        
        <Input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to summarize..."
          className="w-full"
        />
        <Button 
          onClick={testSummarize} 
          disabled={loading || !text.trim()}
          className="w-full"
        >
          Test Summarize Endpoint
        </Button>
      </div>

      {loading && (
        <div className="text-center text-sm text-muted-foreground">
          Testing...
        </div>
      )}

      {error && (
        <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {summary && (
        <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-md">
          <p className="text-sm text-green-700 dark:text-green-400">{summary}</p>
        </div>
      )}
    </div>
  );
};
