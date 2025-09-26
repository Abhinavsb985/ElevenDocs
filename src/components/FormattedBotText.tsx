import React from 'react';

interface FormattedBotTextProps {
  text: string;
  isTyping?: boolean;
  isTypewriterActive?: boolean;
  shouldShowTypewriter?: boolean;
}

export const FormattedBotText = ({ text, isTyping, isTypewriterActive, shouldShowTypewriter = false }: FormattedBotTextProps) => {
  const formatText = (text: string) => {
    // Split text into lines and process them
    const lines = text.split('\n');
    const formattedLines: JSX.Element[] = [];
    let listIndex = 1;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();
      
      // Skip empty lines but add spacing
      if (trimmedLine.length === 0) {
        formattedLines.push(<div key={`spacing-${i}`} className="h-4" />);
        continue;
      }
      
      // Check for headings
      if (trimmedLine.startsWith('# ')) {
        formattedLines.push(
          <h1 key={i} className="text-2xl font-bold text-white mb-6 mt-8 first:mt-0 leading-tight">
            {trimmedLine.substring(2)}
          </h1>
        );
        continue;
      }
      
      if (trimmedLine.startsWith('## ')) {
        formattedLines.push(
          <h2 key={i} className="text-xl font-bold text-white mb-4 mt-6 first:mt-0 leading-tight">
            {trimmedLine.substring(3)}
          </h2>
        );
        continue;
      }
      
      if (trimmedLine.startsWith('### ')) {
        formattedLines.push(
          <h3 key={i} className="text-lg font-bold text-white mb-3 mt-5 first:mt-0 leading-tight">
            {trimmedLine.substring(4)}
          </h3>
        );
        continue;
      }
      
      // Check for bullet points
      if (trimmedLine.startsWith('• ') || trimmedLine.startsWith('- ') || trimmedLine.startsWith('* ')) {
        formattedLines.push(
          <div key={i} className="ml-6 mb-3 flex items-start">
            <span className="text-white mr-3 flex-shrink-0 mt-1 text-lg">•</span>
            <span className="text-base text-white leading-relaxed flex-1">
              {formatInlineText(trimmedLine.substring(2))}
            </span>
          </div>
        );
        continue;
      }
      
      // Check for numbered lists
      if (/^\d+\.\s/.test(trimmedLine)) {
        const match = trimmedLine.match(/^(\d+)\.\s(.+)$/);
        if (match) {
          formattedLines.push(
            <div key={i} className="ml-6 mb-3 flex items-start">
              <span className="text-white mr-3 flex-shrink-0 mt-1 font-medium">
                {match[1]}.
              </span>
              <span className="text-base text-white leading-relaxed flex-1">
                {formatInlineText(match[2])}
              </span>
            </div>
          );
        }
        continue;
      }
      
      // Check for bold text (surrounded by **)
      if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**') && trimmedLine.length > 4) {
        formattedLines.push(
          <p key={i} className="text-base font-bold text-white mb-4 leading-relaxed">
            {trimmedLine.substring(2, trimmedLine.length - 2)}
          </p>
        );
        continue;
      }
      
      // Regular paragraphs
      if (trimmedLine.length > 0) {
        formattedLines.push(
          <p key={i} className="text-base text-white mb-4 leading-relaxed">
            {formatInlineText(trimmedLine)}
          </p>
        );
      }
    }
    
    return formattedLines;
  };

  // Helper function to format inline text (bold, italic, etc.)
  const formatInlineText = (text: string) => {
    // Split by ** for bold text
    const parts = text.split(/(\*\*.*?\*\*)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={index} className="font-bold text-white">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  if (isTyping) {
    return (
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          {[0, 1, 2].map((index) => (
            <div
              key={index}
              className="w-2 h-2 bg-white rounded-full"
              style={{
                animation: `typing-dots 1.4s infinite ${index * 0.2}s`,
              }}
            />
          ))}
        </div>
  <span className="text-base text-gray-300">Verion is analyzing...</span>
      </div>
    );
  }

  return (
    <div className="bot-text-content max-w-none whitespace-pre-wrap">
      <div className="space-y-1">
        {formatText(text)}
        {shouldShowTypewriter && isTypewriterActive && (
          <span className="inline-block w-0.5 h-4 bg-white ml-1 animate-pulse" style={{
            animation: 'typewriter-cursor 0.8s infinite'
          }} />
        )}
      </div>
    </div>
  );
};
