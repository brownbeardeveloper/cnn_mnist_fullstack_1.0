'use client'

import { useState } from 'react'
import Canvas from '@/components/Canvas'
import ConfidenceDisplay from '@/components/ConfidenceDisplay'

interface PredictionResult {
    prediction: number
    probabilities: number[]
    logits: number[]
    min_logit: number
    max_logit: number
}

export default function Home() {
    const [isLoading, setIsLoading] = useState(false)
    const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const handlePredict = async (imageBlob: Blob) => {
        setIsLoading(true)
        setError(null)

        try {
            const formData = new FormData()
            formData.append('file', imageBlob, 'digit.png')

            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            const result = await response.json()

            // Log the raw result for debugging
            console.log('Raw prediction result:', result);

            // Validate the prediction field exists and is a number
            if (typeof result.prediction !== 'number' && result.prediction !== 0) {
                console.error('Invalid prediction value:', result.prediction);
                setError(`Invalid prediction value: ${result.prediction}`);
            } else {
                // Ensure prediction is properly set as a number (even if it's 0)
                const validatedResult: PredictionResult = {
                    ...result,
                    prediction: Number(result.prediction),
                }
                setPredictionResult(validatedResult)
            }
        } catch (error) {
            console.error('Error predicting digit:', error)
            setError(error instanceof Error ? error.message : 'Unknown error occurred')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <main className="py-8">
            <div className="max-w-2xl mx-auto p-4">
                <div className="bg-white rounded-lg shadow-sm overflow-hidden">
                    <h1 className="text-2xl font-bold text-center p-4 bg-gray-800 text-white">
                        MNIST Digit Recognizer
                    </h1>

                    {error && (
                        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 m-4 rounded">
                            <p>{error}</p>
                        </div>
                    )}

                    <div className="p-6">
                        <div className="flex flex-col md:flex-row md:space-x-8">
                            <div className="w-full md:w-1/2">
                                <Canvas onPredict={handlePredict} isLoading={isLoading} />
                            </div>

                            <div className="w-full md:w-1/2 mt-8 md:mt-0">
                                <ConfidenceDisplay
                                    prediction={predictionResult?.prediction ?? null}
                                    probabilities={predictionResult?.probabilities ?? null}
                                    logits={predictionResult?.logits ?? null}
                                    minLogit={predictionResult?.min_logit ?? null}
                                    maxLogit={predictionResult?.max_logit ?? null}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    )
} 