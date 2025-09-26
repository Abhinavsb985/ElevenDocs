import { useState, useRef } from "react";
import { Paperclip, Send } from "lucide-react";

interface ProgrammaticSearchBarProps {
  onSearch?: (query: string, files?: File[]) => void;
  placeholder?: string;
  className?: string;
  apiEndpoint?: string;
}

export const ProgrammaticSearchBar = ({ 
  onSearch,
  placeholder = "Ask ElevenDocs or upload...",
  className = "",
  apiEndpoint = "http://127.0.0.1:8002/invoke"
}: ProgrammaticSearchBarProps) => {
  const [query, setQuery] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    const newFiles = Array.from(files);
    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && uploadedFiles.length === 0) return;
    
    setIsLoading(true);
    
    try {
      let response;
      
      if (uploadedFiles.length > 0) {
        // Handle file upload
        const formData = new FormData();
        formData.append('input', query || 'Please analyze this document');
        
        // Process each uploaded file
        for (const uploadedFile of uploadedFiles) {
          const fileType = uploadedFile.type.split('/')[1] || 
                          uploadedFile.name.split('.').pop()?.toLowerCase() || 'txt';
          formData.append('file_data', uploadedFile);
          formData.append('file_type', fileType);
        }

        response = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData
        });
      } else {
        // Handle text-only message
        response = await fetch(apiEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: query })
        });
      }

      if (!response.ok) {
        throw new Error('Request failed');
      }

      const data = await response.json();
      const result = data.response || data.output || 'No response from agent';
      
      // Call the onSearch callback with the result
      if (onSearch) {
        onSearch(query, uploadedFiles, result);
      } else {
        // Default behavior: show result in alert
        alert(`AI Response: ${result}`);
      }
      
    } catch (error) {
      console.error('Search error:', error);
      if (onSearch) {
        onSearch(query, uploadedFiles, 'Error: Unable to process your request. Please try again.');
      } else {
        alert('Error: Unable to process your request. Please try again.');
      }
    } finally {
      setIsLoading(false);
      setQuery("");
      setUploadedFiles([]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={`w-full max-w-4xl mx-auto ${className}`}>
      {/* File Preview Area */}
      {uploadedFiles.length > 0 && (
        <div className="mb-4 p-4 bg-gray-900 border border-gray-700 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm font-medium text-gray-300">Selected Files:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-2 bg-gray-800 border border-gray-600 px-3 py-2 rounded-md text-sm"
              >
                <span className="truncate max-w-[200px] text-gray-200">{file.name}</span>
                <span className="text-gray-400 text-xs">
                  ({(file.size / 1024).toFixed(1)} KB)
                </span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-500 hover:text-red-400 text-lg leading-none"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search Container */}
      <form onSubmit={handleSubmit} className="w-full">
        <div className="search-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            className="search-bar"
            disabled={isLoading}
          />
          <button
            type="button"
            onClick={triggerFileInput}
            className="icon-btn"
            title="Attach file"
            disabled={isLoading}
          >
            <Paperclip size={20} />
          </button>
          <button
            type="submit"
            className="icon-btn"
            disabled={(!query.trim() && uploadedFiles.length === 0) || isLoading}
          >
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
          onChange={handleFileSelect}
          className="hidden"
        />
      </form>
    </div>
  );
};
