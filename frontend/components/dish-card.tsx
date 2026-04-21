'use client'

import { useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, ChevronDown, ChevronUp, MapPin } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { DishRecommendation, Dish } from '@/app/page'

// Unsplash photo IDs mapped by cuisine type.
// These are stable direct image URLs — no API key needed.
const CUISINE_IMAGES: Record<string, string> = {
  'north-indian':   'https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&q=75',
  'south-indian':   'https://images.unsplash.com/photo-1630383249896-424e482df921?w=600&q=75',
  'chinese':        'https://images.unsplash.com/photo-1563245372-f21724e3856d?w=600&q=75',
  'continental':    'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&q=75',
  'street-food':    'https://images.unsplash.com/photo-1606491956391-6a5e2b9c7e58?w=600&q=75',
  'mughlai':        'https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?w=600&q=75',
  'italian':        'https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&q=75',
  'fast-food':      'https://images.unsplash.com/photo-1550547660-d9450f859349?w=600&q=75',
  'default':        'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&q=75',
}

function getImageUrl(cuisineType: string | string[] | undefined): string {
  const raw = Array.isArray(cuisineType) ? cuisineType[0] : cuisineType
  const key = raw?.toLowerCase().replace(/\s+/g, '-') ?? 'default'
  return CUISINE_IMAGES[key] ?? CUISINE_IMAGES['default']
}

// ─── Veg / Non-veg dot ───────────────────────────────────────────────────────

function VegIndicator({ vegType }: { vegType: string }) {
  const bgColor = vegType === 'veg' ? '#21C55D' : vegType === 'contains-egg' ? '#F59E0B' : '#EF4444'
  const borderColor = vegType === 'veg' ? '#16A34A' : vegType === 'contains-egg' ? '#D97706' : '#DC2626'
  const label = vegType === 'veg' ? 'Vegetarian' : vegType === 'contains-egg' ? 'Contains Egg' : 'Non-vegetarian'

  return (
    <div
      className="w-4 h-4 rounded flex-shrink-0 border-2"
      style={{ backgroundColor: bgColor, borderColor }}
      title={label}
    />
  )
}

// ─── Similar dish thumbnail ───────────────────────────────────────────────────

function SimilarDishCard({ dish }: { dish: Dish }) {
  const [imgError, setImgError] = useState(false)
  const imgUrl = getImageUrl(dish.cuisineType)

  return (
    <div className="flex-shrink-0 w-20 bg-white rounded-lg overflow-hidden shadow-sm border border-border hover:shadow-md transition-shadow">
      <div className="w-full h-12 overflow-hidden bg-orange-50">
        {!imgError ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imgUrl}
            alt={dish.name}
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
            🍽
          </div>
        )}
      </div>
      <div className="p-1.5">
        <p className="text-xs text-center text-foreground font-medium line-clamp-2">
          {dish.name}
        </p>
      </div>
    </div>
  )
}

// ─── Restaurant list (drill-down) ────────────────────────────────────────────

function RestaurantList({ restaurants }: { restaurants: string[] }) {
  return (
    <div className="mt-3 space-y-1.5">
      {restaurants.map((r) => (
        <div key={r} className="flex items-center gap-2 py-1.5 px-2 bg-gray-50 rounded-lg">
          <MapPin className="h-3 w-3 text-primary flex-shrink-0" />
          <span className="text-xs text-foreground">{r}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Main card ───────────────────────────────────────────────────────────────

interface DishCardProps {
  dish: DishRecommendation
}

export function DishCard({ dish }: DishCardProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showLeftArrow, setShowLeftArrow]       = useState(false)
  const [showRightArrow, setShowRightArrow]     = useState(true)
  const [showRestaurants, setShowRestaurants]   = useState(false)
  const [imgError, setImgError]                 = useState(false)

  const imgUrl = getImageUrl(dish.cuisineType)

  const scroll = (direction: 'left' | 'right') => {
    scrollRef.current?.scrollBy({
      left: direction === 'left' ? -240 : 240,
      behavior: 'smooth',
    })
  }

  const handleScroll = () => {
    if (!scrollRef.current) return
    setShowLeftArrow(scrollRef.current.scrollLeft > 0)
    setShowRightArrow(
      scrollRef.current.scrollLeft <
      scrollRef.current.scrollWidth - scrollRef.current.clientWidth - 10
    )
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm overflow-hidden border border-border">

      {/* Hero image */}
      <div className="w-full h-40 relative overflow-hidden bg-orange-50">
        {!imgError ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imgUrl}
            alt={dish.name}
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          <div
            className="w-full h-full flex items-center justify-center text-muted-foreground"
            style={{ background: 'linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%)' }}
          >
            <span className="text-4xl">🍽</span>
          </div>
        )}
      </div>

      {/* Card body */}
      <div className="p-4">

        {/* Dish name + veg dot */}
        <div className="flex gap-2 items-start mb-2">
          <VegIndicator vegType={dish.vegType} />
          <h2 className="text-base font-bold text-foreground">{dish.name}</h2>
        </div>

        {/* Availability — tappable to expand restaurant list */}
        <button
          onClick={() => dish.availableAt > 1 ? setShowRestaurants((v) => !v) : undefined}
          className={`flex items-center gap-1.5 mb-2 ${dish.availableAt > 1 ? 'group cursor-pointer' : 'cursor-default'}`}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-primary" />
          <p className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
            {dish.availableAt === 1
              ? `At ${dish.allRestaurants?.[0] ?? 'this restaurant'}`
              : `Available at ${dish.availableAt} restaurants near you`}
          </p>
          {dish.availableAt > 1 && (showRestaurants
            ? <ChevronUp className="h-3 w-3 text-muted-foreground" />
            : <ChevronDown className="h-3 w-3 text-muted-foreground" />
          )}
        </button>

        {/* Restaurant drill-down */}
        {showRestaurants && dish.allRestaurants.length > 0 && (
          <RestaurantList restaurants={dish.allRestaurants} />
        )}

        {/* Price */}
        <p className="text-sm font-semibold text-foreground mt-2 mb-4">
          {dish.priceMin === dish.priceMax
            ? `₹${dish.priceMin}`
            : `₹${dish.priceMin} – ₹${dish.priceMax}`}
        </p>

        {/* Divider */}
        <div className="h-px bg-border mb-4" />

        {/* Similar dishes */}
        {dish.similarDishes.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-muted-foreground uppercase mb-3 tracking-wide">
              Similar dishes
            </h3>
            <div className="relative">
              <div
                ref={scrollRef}
                onScroll={handleScroll}
                className="flex gap-2 overflow-x-auto scrollbar-hide pb-2"
              >
                {dish.similarDishes.map((s) => (
                  <SimilarDishCard key={s.id} dish={s} />
                ))}
              </div>

              {showLeftArrow && dish.similarDishes.length >= 4 && (
                <Button
                  size="icon" variant="ghost"
                  className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1 h-7 w-7 rounded-full bg-white border border-border z-10"
                  onClick={() => scroll('left')}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
              )}
              {showRightArrow && dish.similarDishes.length >= 4 && (
                <Button
                  size="icon" variant="ghost"
                  className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1 h-7 w-7 rounded-full bg-white border border-border z-10"
                  onClick={() => scroll('right')}
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
