import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const suggestions = [
  {
    id: 1,
    title: "Evaluate arguments",
    subtitle: "for universal healthcare",
  },
  {
    id: 2,
    title: "Analyze consequences",
    subtitle: "of space exploration",
  },
  {
    id: 3,
    title: "Generate Docker Dockerize",
    subtitle: "for data analysis a Python ML ...",
  },
];

export const SuggestionCards = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl">
      {suggestions.map((suggestion, index) => (
        <Card
          key={suggestion.id}
          className={cn(
            "p-4 bg-suggestion-card hover:bg-suggestion-card-hover border border-border cursor-pointer transition-all duration-300 group ai-float slide-up hover:shadow-ai hover:border-primary/30",
          )}
          style={{ 
            animationDelay: `${300 + index * 100}ms`,
          }}
        >
          <div className="space-y-1">
            <h3 className="font-medium text-foreground group-hover:text-primary transition-colors">
              {suggestion.title}
            </h3>
            <p className="text-sm text-muted-foreground">
              {suggestion.subtitle}
            </p>
          </div>
        </Card>
      ))}
    </div>
  );
};