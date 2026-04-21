interface EmptyStateProps {
  query?: string
  onExampleClick?: (example: string) => void
}

export function EmptyState({ query, onExampleClick }: EmptyStateProps) {
  const examples = ['spicy street food under ₹200', 'not feeling well', 'cheat day', 'creamy mild curry', 'surprise me']

  if (query) {
    return (
      <div className="text-center py-12">
        <p className="text-base font-medium text-foreground mb-2">
          No matches for &quot;{query}&quot;
        </p>
        <p className="text-sm text-muted-foreground">
          Try searching for different dishes
        </p>
      </div>
    )
  }

  return (
    <div className="text-center py-12">
      <h2 className="text-xl font-bold text-foreground mb-4">
        What are you craving today?
      </h2>
      <div className="flex flex-wrap gap-2 justify-center">
        {examples.map((example) => (
          <button
            key={example}
            onClick={() => onExampleClick?.(example)}
            className="px-4 py-2 rounded-full border border-primary text-primary text-sm hover:bg-primary hover:text-primary-foreground transition-all cursor-pointer"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  )
}
