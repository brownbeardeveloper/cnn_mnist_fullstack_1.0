'use client'

interface ConfidenceDisplayProps {
    prediction: number | null
    probabilities: number[] | null
    logits: number[] | null
    minLogit: number | null
    maxLogit: number | null
}

export default function ConfidenceDisplay({
    prediction,
    probabilities,
    logits,
    minLogit,
    maxLogit
}: ConfidenceDisplayProps) {
    // Debug information about the prediction value
    const predictionType = prediction === null ? 'null' :
        prediction === undefined ? 'undefined' :
            typeof prediction;

    if (prediction === null || prediction === undefined) {
        return (
            <div className="flex flex-col items-center justify-center h-full">
                <p className="text-gray-500 text-center">
                    Draw a digit and click "Recognize" to see predictions
                </p>
                <p className="text-xs text-gray-400 mt-2">
                    Debug: prediction is {predictionType}
                </p>
            </div>
        )
    }

    return (
        <div className="flex flex-col">
            <div className="mb-4 text-center">
                <h3 className="text-2xl font-bold">Prediction: {prediction}</h3>
                <p className="text-xs text-gray-400">
                    (Type: {predictionType}, Value: {JSON.stringify(prediction)})
                </p>
            </div>

            <div className="space-y-2">
                <h4 className="font-medium text-gray-700">Confidence</h4>
                <div className="space-y-1">
                    {probabilities?.map((prob, i) => (
                        <div key={i} className="flex items-center">
                            <span className="w-8 text-gray-700">{i}:</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-5 overflow-hidden">
                                <div
                                    className={`h-full ${i === prediction ? 'bg-blue-500' : 'bg-gray-500'
                                        }`}
                                    style={{ width: `${prob * 100}%` }}
                                />
                            </div>
                            <span className="w-14 text-right text-gray-700 ml-2">
                                {(prob * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
} 