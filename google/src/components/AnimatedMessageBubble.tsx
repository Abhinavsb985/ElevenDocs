import { motion } from 'framer-motion';
import { User, Bot, FileText } from "lucide-react";
import { useTypewriter } from '@/hooks/useTypewriter';
import { FormattedBotText } from './FormattedBotText';

interface AnimatedMessageBubbleProps {
  message: string;
  isUser: boolean;
  timestamp: string;
  fileInfo?: {
    name: string;
    type: string;
    size: number;
  };
  isTyping?: boolean;
  showTypewriter?: boolean;
  onTypewriterComplete?: () => void;
  isLatestBotMessage?: boolean;
}

export const AnimatedMessageBubble = ({ 
  message, 
  isUser, 
  timestamp, 
  fileInfo,
  isTyping = false,
  showTypewriter = false,
  onTypewriterComplete,
  isLatestBotMessage = false
}: AnimatedMessageBubbleProps) => {
  const { displayedText, isTyping: isTypewriterActive } = useTypewriter({
    text: (showTypewriter && isLatestBotMessage) ? message : '',
    speed: 30,
    onComplete: onTypewriterComplete
  });

  const messageVariants = {
    hidden: { opacity: 0, y: 10, scale: 0.98 },
    visible: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        duration: 0.4,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    }
  };

  const bubbleVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.3,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    }
  };

  return (
    <motion.div 
      className={`flex gap-3 px-4 py-6 ${isUser ? 'justify-end' : 'justify-start'}`}
      variants={messageVariants}
      initial="hidden"
      animate="visible"
    >
      {!isUser && (
        <motion.div 
          className="flex-shrink-0 w-8 h-8 bg-ai-message border border-border rounded-full flex items-center justify-center"
          variants={bubbleVariants}
        >
          <Bot size={16} className="text-ai-message-foreground" />
        </motion.div>
      )}
      
      <div className={`max-w-[90%] sm:max-w-[85%] md:max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        <motion.div 
          className={`${
            isUser 
              ? 'bg-blue-500 text-white rounded-2xl px-4 py-3 ml-auto' 
              : 'text-white'
          }`}
          variants={bubbleVariants}
        >
          {fileInfo && (
            <div className="flex items-center gap-2 mb-2 p-2 bg-muted/50 rounded-md">
              <FileText size={14} className="text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                {fileInfo.name} ({(fileInfo.size / 1024).toFixed(1)} KB)
              </span>
            </div>
          )}
          
          <motion.div
            key={isTyping ? 'typing' : showTypewriter ? 'typewriter' : 'static'}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            {isUser ? (
              <p className="text-base leading-relaxed whitespace-pre-wrap">{message}</p>
            ) : (
              <FormattedBotText 
                text={showTypewriter ? displayedText : message}
                isTyping={isTyping}
                isTypewriterActive={isTypewriterActive}
                shouldShowTypewriter={showTypewriter && isLatestBotMessage}
              />
            )}
          </motion.div>
        </motion.div>
        
        <motion.p 
          className={`text-xs text-muted-foreground mt-1 ${isUser ? 'text-right' : 'text-left'}`}
          variants={bubbleVariants}
        >
          {timestamp}
        </motion.p>
      </div>
      
      {isUser && (
        <motion.div 
          className="flex-shrink-0 w-8 h-8 bg-user-message border border-border rounded-full flex items-center justify-center"
          variants={bubbleVariants}
        >
          <User size={16} className="text-user-message-foreground" />
        </motion.div>
      )}
    </motion.div>
  );
};
