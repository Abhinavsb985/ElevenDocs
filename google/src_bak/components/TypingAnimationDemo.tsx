import { useState } from 'react';
import { AnimatedMessageBubble } from './AnimatedMessageBubble';
import { Button } from '@/components/ui/button';

export const TypingAnimationDemo = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! This is a demo of the typing animation.",
      isUser: false,
      timestamp: "Just now",
      isTyping: false,
      showTypewriter: false
    }
  ]);

  const addTypingMessage = () => {
    const newMessage = {
      id: messages.length + 1,
      text: "",
      isUser: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isTyping: true,
      showTypewriter: false
    };
    
    setMessages(prev => [...prev, newMessage]);
    
    // Simulate API response after 2 seconds
    setTimeout(() => {
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => !msg.isTyping);
        return [...withoutTyping, {
          id: newMessage.id,
          text: `# Legal Document Analysis

## Summary
This is a comprehensive analysis of your legal document. Here are the key findings:

The document appears to be a **standard service agreement** with typical clauses and conditions. However, there are several important points that require your attention.

### Important Points
• **Contract Terms**: The agreement includes standard clauses that are generally acceptable
• **Payment Schedule**: Monthly payments of $500 with automatic renewal
• **Termination Clause**: 30-day notice required for cancellation
• **Liability Limitations**: Standard liability caps are in place

### Risk Assessment
1. **Low Risk**: Standard legal language and boilerplate clauses
2. **Medium Risk**: Payment terms may need review for your specific situation
3. **High Risk**: Termination clause requires careful attention

### Key Recommendations
• Review the **payment terms** carefully
• Understand the **termination process** before signing
• Consider consulting with a legal professional

**Important**: Always consult with a qualified attorney before signing any legal document.`,
          isUser: false,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isTyping: false,
          showTypewriter: true
        }];
      });
    }, 2000);
  };

  const addUserMessage = () => {
    const newMessage = {
      id: messages.length + 1,
      text: "This is a user message that appears instantly!",
      isUser: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isTyping: false,
      showTypewriter: false
    };
    
    setMessages(prev => [...prev, newMessage]);
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex gap-2">
        <Button onClick={addUserMessage} variant="outline">
          Add User Message
        </Button>
        <Button onClick={addTypingMessage}>
          Test Typing Animation
        </Button>
      </div>
      
      <div className="space-y-2">
        {messages.map((message, index) => {
          // Check if this is the latest bot message that should show typing animation
          const isLatestBotMessage = !message.isUser && 
            messages.slice(index + 1).every(msg => msg.isUser || !msg.showTypewriter) &&
            message.showTypewriter;
          
          return (
            <AnimatedMessageBubble
              key={message.id}
              message={message.text}
              isUser={message.isUser}
              timestamp={message.timestamp}
              isTyping={message.isTyping}
              showTypewriter={message.showTypewriter}
              isLatestBotMessage={isLatestBotMessage}
            />
          );
        })}
      </div>
    </div>
  );
};
