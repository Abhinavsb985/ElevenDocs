import { useAuth } from '@/contexts/AuthContext';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const useApi = () => {
  const { idToken, logout } = useAuth();

  const makeRequest = async (
    endpoint: string,
    options: RequestInit = {}
  ) => {
    if (!idToken) {
      throw new Error('No authentication token available');
    }

    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${idToken}`,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (response.status === 401) {
        // Token expired or invalid
        logout();
        throw new Error('Authentication failed. Please sign in again.');
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  };

  const get = (endpoint: string) => makeRequest(endpoint, { method: 'GET' });
  
  const post = (endpoint: string, data: any) => 
    makeRequest(endpoint, { 
      method: 'POST', 
      body: JSON.stringify(data) 
    });
  
  const put = (endpoint: string, data: any) => 
    makeRequest(endpoint, { 
      method: 'PUT', 
      body: JSON.stringify(data) 
    });
  
  const del = (endpoint: string) => 
    makeRequest(endpoint, { method: 'DELETE' });

  return {
    get,
    post,
    put,
    delete: del,
    makeRequest
  };
};
