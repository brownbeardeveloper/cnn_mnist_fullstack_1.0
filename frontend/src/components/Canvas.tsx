'use client'

import { useRef, useState, useEffect } from 'react'

interface CanvasProps {
    onPredict: (imageBlob: Blob) => Promise<void>
    isLoading: boolean
}

export default function Canvas({ onPredict, isLoading }: CanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [isDrawing, setIsDrawing] = useState(false)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Set up canvas
        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 15
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
    }, [])

    const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        setIsDrawing(true)

        // Get position
        let x, y
        if ('touches' in e) {
            // Touch event
            const rect = canvas.getBoundingClientRect()
            x = e.touches[0].clientX - rect.left
            y = e.touches[0].clientY - rect.top
        } else {
            // Mouse event
            x = e.nativeEvent.offsetX
            y = e.nativeEvent.offsetY
        }

        ctx.beginPath()
        ctx.moveTo(x, y)
    }

    const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
        if (!isDrawing) return

        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Get position
        let x, y
        if ('touches' in e) {
            // Touch event
            const rect = canvas.getBoundingClientRect()
            x = e.touches[0].clientX - rect.left
            y = e.touches[0].clientY - rect.top
        } else {
            // Mouse event
            x = e.nativeEvent.offsetX
            y = e.nativeEvent.offsetY
        }

        ctx.lineTo(x, y)
        ctx.stroke()
    }

    const stopDrawing = () => {
        setIsDrawing(false)
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }

    const handlePredict = async () => {
        const canvas = canvasRef.current
        if (!canvas) return

        // Get the image data
        const imageBlob = await new Promise<Blob>((resolve) => {
            canvas.toBlob((blob) => {
                if (blob) resolve(blob)
            })
        })

        await onPredict(imageBlob)
    }

    return (
        <div className="flex flex-col items-center">
            <canvas
                ref={canvasRef}
                width={280}
                height={280}
                className="border border-gray-300 touch-none"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
            />
            <div className="flex mt-4 gap-4">
                <button
                    onClick={clearCanvas}
                    className="px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300 transition-colors"
                    disabled={isLoading}
                >
                    Clear
                </button>
                <button
                    onClick={handlePredict}
                    className={`px-4 py-2 rounded transition-colors ${isLoading
                        ? 'bg-blue-300 cursor-not-allowed'
                        : 'bg-blue-500 text-white hover:bg-blue-600'
                        }`}
                    disabled={isLoading}
                >
                    {isLoading ? 'Recognizing...' : 'Recognize'}
                </button>
            </div>
        </div>
    )
} 