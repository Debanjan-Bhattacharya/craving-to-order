'use client'

import { useState } from 'react'
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { DishCard } from '@/components/dish-card'
import { EmptyState } from '@/components/empty-state'
import { LoadingSkeleton } from '@/components/loading-skeleton'

type VegType = 'veg' | 'non-veg' | 'contains-egg' | string

export interface Dish {
  id: string
  name: string
  vegType: VegType
  priceMin: number
  priceMax: number
  availableAt: number
  cuisineType: string | string[] | undefined
}

export interface DishRecommendation extends Dish {
  allRestaurants: string[]
  similarDishes: Dish[]
}

// Raw hit shape from /recommend API
interface ApiHit {
  dish: string
  restaurant: string
  price: number
  tags: string[]
  cuisine_type: string
  cuisine: string
  rerank_score: number
  restaurant_count: number
  all_restaurants: string[]
  similar_dishes?: ApiHit[]
}

const API_URL = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/recommend`
  : 'http://127.0.0.1:8000/recommend'

function mapHitToDish(hit: ApiHit, index: number): Dish {
  return {
    id: String(index),
    name: hit.dish,
    vegType: hit.is_veg ?? (hit.tags?.includes('non-veg') ? 'non-veg' : hit.tags?.includes('contains-egg') ? 'contains-egg' : 'veg'),
    priceMin: hit.price_min ?? hit.price,
    priceMax: hit.price_max ?? hit.price,
    availableAt: hit.restaurant_count ?? 1,
    cuisineType: hit.cuisine_type ?? hit.cuisine ?? 'default',
  }
}

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [dishes, setDishes] = useState<DishRecommendation[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [apiResponse, setApiResponse] = useState<string | null>(null)

  const handleSearch = async (query: string = searchQuery) => {
    if (!query.trim()) return

    setIsLoading(true)
    setHasSearched(true)
    setApiError(null)

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      if (!res.ok) throw new Error(`API returned ${res.status}`)

      const data = await res.json()
      const hits: ApiHit[] = data.hits ?? []

      // Map each hit; use the other hits (up to 3) as similar dishes
      const mapped: DishRecommendation[] = hits.map((hit, i) => ({
        ...mapHitToDish(hit, i),
        allRestaurants: hit.all_restaurants ?? [hit.restaurant],
        similarDishes: (hit.similar_dishes ?? []).map((s: ApiHit, si: number) =>
  mapHitToDish(s, Number(`${i}${si}`))),
      }))

      setDishes(mapped)
      setApiResponse(data.response ?? null)
    } catch (err) {
      console.error('API error:', err)
      setApiError('Could not reach the backend. Is FastAPI running on port 8000?')
      setDishes([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleExampleClick = (example: string) => {
    setSearchQuery(example)
    handleSearch(example)
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="max-w-md mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-foreground mb-1">
            Craving to Order
          </h1>
          <p className="text-sm text-muted-foreground">
            Tell us what you want, we&apos;ll find it for you
          </p>
        </div>

        {/* Search Bar */}
        <form
          onSubmit={(e) => { e.preventDefault(); handleSearch() }}
          className="mb-8"
        >
          <div className="flex gap-2">
            <Input
              type="text"
              placeholder="Tell me what you're craving..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-white border-border text-foreground placeholder:text-muted-foreground"
            />
            <Button
              type="submit"
              size="icon"
              className="bg-primary hover:bg-primary/90 text-primary-foreground"
              disabled={isLoading}
            >
              {isLoading ? (
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
              ) : (
                <Search className="h-5 w-5" />
              )}
            </Button>
          </div>
        </form>

        {/* API Error */}
        {apiError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {apiError}
          </div>
        )}

        {/* Content */}
        {!hasSearched ? (
          <EmptyState onExampleClick={handleExampleClick} />
        ) : isLoading ? (
          <LoadingSkeleton />
        ) : dishes.length > 0 ? (
          <div>
            <p className="text-sm text-muted-foreground mb-4">
              {dishes.length} {dishes.length === 1 ? 'dish' : 'dishes'} matching your craving
            </p>
            <div className="space-y-6">
              {dishes.map((dish) => (
                <DishCard key={dish.id} dish={dish} />
              ))}
            </div>
          </div>
        ) : apiResponse ? (
  <div className="text-sm text-muted-foreground text-center py-8">{apiResponse}</div>
) : (
  <EmptyState query={searchQuery} />
)}
      </div>
    </main>
  )
}
