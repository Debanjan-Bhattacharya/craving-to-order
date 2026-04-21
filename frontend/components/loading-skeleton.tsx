'use client'

export function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-white rounded-2xl shadow-sm overflow-hidden">
          {/* Image skeleton */}
          <div className="w-full h-40 bg-gradient-to-r from-gray-200 to-gray-100 animate-pulse" />

          {/* Content skeleton */}
          <div className="p-4 space-y-3">
            <div className="flex gap-2 items-start">
              <div className="w-4 h-4 rounded bg-gray-200 animate-pulse flex-shrink-0 mt-1" />
              <div className="flex-1">
                <div className="h-5 bg-gray-200 rounded animate-pulse w-3/4 mb-2" />
                <div className="h-4 bg-gray-100 rounded animate-pulse w-1/2" />
              </div>
            </div>

            <div className="h-4 bg-gray-100 rounded animate-pulse w-2/3" />
            <div className="h-4 bg-gray-100 rounded animate-pulse w-1/2" />

            {/* Similar dishes skeleton */}
            <div className="pt-4 border-t border-gray-100">
              <div className="h-3 bg-gray-200 rounded animate-pulse w-1/4 mb-3" />
              <div className="flex gap-3">
                {[1, 2, 3].map((j) => (
                  <div key={j} className="flex-shrink-0">
                    <div className="w-20 h-28 bg-gray-200 rounded animate-pulse" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
