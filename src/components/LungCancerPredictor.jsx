import React, { useState, useEffect } from 'react';
import { Upload, Activity, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';
import Papa from 'papaparse';

const LungCancerPredictor = () => {
  const [model, setModel] = useState(null);
  const [modelStats, setModelStats] = useState(null);
  const [formData, setFormData] = useState({
    edad: 50,
    anos_fumando: 0,
    humo_segunda_mano: 0,
    historial_familiar: 0,
    exposicion_quimicos: 0,
    enfermedad_pulmonar: 0,
    tos_persistente: 0,
    dificultad_respirar: 0,
    dolor_pecho: 0
  });
  const [prediction, setPrediction] = useState(null);
  const [trainingData, setTrainingData] = useState([]);
  const [isTraining, setIsTraining] = useState(false);

  // ================================
  //       DATASET POR DEFECTO
  // ================================
  const defaultDataset = [
    // positivos
    { edad: 65, anos_fumando: 40, humo_segunda_mano: 1, historial_familiar: 1, exposicion_quimicos: 1, enfermedad_pulmonar: 1, tos_persistente: 1, dificultad_respirar: 1, dolor_pecho: 1, cancer: 1 },
    { edad: 70, anos_fumando: 45, humo_segunda_mano: 1, historial_familiar: 1, exposicion_quimicos: 0, enfermedad_pulmonar: 1, tos_persistente: 1, dificultad_respirar: 1, dolor_pecho: 0, cancer: 1 },
    { edad: 62, anos_fumando: 35, humo_segunda_mano: 1, historial_familiar: 0, exposicion_quimicos: 1, enfermedad_pulmonar: 1, tos_persistente: 1, dificultad_respirar: 1, dolor_pecho: 1, cancer: 1 },
    // negativos
    { edad: 30, anos_fumando: 0, humo_segunda_mano: 0, historial_familiar: 0, exposicion_quimicos: 0, enfermedad_pulmonar: 0, tos_persistente: 0, dificultad_respirar: 0, dolor_pecho: 0, cancer: 0 },
    { edad: 35, anos_fumando: 5, humo_segunda_mano: 0, historial_familiar: 0, exposicion_quimicos: 0, enfermedad_pulmonar: 0, tos_persistente: 0, dificultad_respirar: 0, dolor_pecho: 0, cancer: 0 },
    { edad: 40, anos_fumando: 0, humo_segunda_mano: 1, historial_familiar: 0, exposicion_quimicos: 0, enfermedad_pulmonar: 0, tos_persistente: 0, dificultad_respirar: 0, dolor_pecho: 0, cancer: 0 }
  ];

  // ================================
  //       SIGMOIDE
  // ================================
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  // ================================
  //   NORMALIZACIÓN CORREGIDA
  // ================================
  const normalize = (data) => {
    const features = Object.keys(formData);

    const stats = {};
    features.forEach(f => {
      const values = data.map(d => d[f]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance =
        values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance) || 1;

      stats[f] = { mean, std };
    });

    const normalized = data.map(row => {
      const nr = { ...row };
      features.forEach(f => {
        nr[f] = (row[f] - stats[f].mean) / stats[f].std;
      });
      return nr;
    });

    return { normalized, stats };
  };

  // ================================
  //     ENTRENAR MODELO
  // ================================
  const trainModel = (data) => {
    setIsTraining(true);

    setTimeout(() => {
      const { normalized, stats } = normalize(data);

      // inicialización de pesos
      const weights = {};
      Object.keys(formData).forEach(k => (weights[k] = Math.random() * 0.2 - 0.1));
      weights.bias = 0;

      const lr = 0.05;
      const epochs = 1200;

      for (let ep = 0; ep < epochs; ep++) {
        const grad = {};
        Object.keys(weights).forEach(k => (grad[k] = 0));

        normalized.forEach(row => {
          const z =
            Object.keys(formData).reduce((sum, k) => sum + weights[k] * row[k], 0) +
            weights.bias;

          const pred = sigmoid(z);
          const error = pred - row.cancer;

          Object.keys(formData).forEach(k => {
            grad[k] += error * row[k];
          });
          grad.bias += error;
        });

        Object.keys(weights).forEach(k => {
          weights[k] -= (lr / normalized.length) * grad[k];
        });
      }

      // precisión
      let correct = 0;
      normalized.forEach(row => {
        const z =
          Object.keys(formData).reduce((sum, k) => sum + weights[k] * row[k], 0) +
          weights.bias;
        const pred = sigmoid(z) >= 0.5 ? 1 : 0;
        if (pred === row.cancer) correct++;
      });

      const accuracy = ((correct / normalized.length) * 100).toFixed(1);

      setModel({ weights, stats });
      setModelStats({
        accuracy,
        samples: data.length,
        positive: data.filter(d => d.cancer === 1).length,
        negative: data.filter(d => d.cancer === 0).length
      });

      setIsTraining(false);
    }, 400);
  };

  // ================================
  //           PREDECIR
  // ================================
  const predict = () => {
    if (!model) return;

    const normalized = {};
    Object.keys(formData).forEach(key => {
      normalized[key] = (formData[key] - model.stats[key].mean) / model.stats[key].std;
    });

    const z =
      Object.keys(formData).reduce((sum, key) => sum + model.weights[key] * normalized[key], 0) +
      model.weights.bias;

    const prob = sigmoid(z);
    const risk = prob >= 0.5;

    setPrediction({
      risk,
      probability: (prob * 100).toFixed(1),
      confidence: risk ? prob : 1 - prob
    });
  };

  // ================================
  //     IMPORTAR CSV
  // ================================
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => {
        const data = res.data.map(r => ({
          edad: Number(r.edad) || 0,
          anos_fumando: Number(r.anos_fumando) || 0,
          humo_segunda_mano: Number(r.humo_segunda_mano) || 0,
          historial_familiar: Number(r.historial_familiar) || 0,
          exposicion_quimicos: Number(r.exposicion_quimicos) || 0,
          enfermedad_pulmonar: Number(r.enfermedad_pulmonar) || 0,
          tos_persistente: Number(r.tos_persistente) || 0,
          dificultad_respirar: Number(r.dificultad_respirar) || 0,
          dolor_pecho: Number(r.dolor_pecho) || 0,
          cancer: Number(r.cancer) || 0
        }));

        setTrainingData(data);
        trainModel(data);
      }
    });
  };

  useEffect(() => {
    setTrainingData(defaultDataset);
    trainModel(defaultDataset);
  }, []);

  // ================================
  //         INTERFAZ
  // ================================
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">🫁 Predictor de Cáncer de Pulmón</h1>
          <p className="text-gray-600">Regresión Logística con Gradiente Descendente</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* PANEL ENTRENAMIENTO */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="text-indigo-600" />
              <h2 className="text-xl font-bold">Entrenamiento del Modelo</h2>
            </div>

            <label className="block text-sm font-medium text-gray-700 mb-2">
              Importar Dataset (CSV)
            </label>

            <label className="cursor-pointer">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 hover:border-indigo-500 transition">
                <div className="flex items-center justify-center gap-2 text-gray-600">
                  <Upload size={20} />
                  <span>Seleccionar archivo CSV</span>
                </div>
              </div>
              <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
            </label>

            {modelStats && (
              <div className="mt-4 bg-indigo-50 p-4 rounded-lg">
                <h3 className="font-semibold">Estadísticas</h3>
                <p>Precisión: {modelStats.accuracy}%</p>
                <p>Muestras: {modelStats.samples}</p>
                <p>Positivos: {modelStats.positive}</p>
                <p>Negativos: {modelStats.negative}</p>
              </div>
            )}

            {isTraining && (
              <div className="flex items-center gap-2 text-indigo-600 mt-4">
                <Activity className="animate-spin" />
                <span>Entrenando...</span>
              </div>
            )}
          </div>

          {/* PANEL PREDICCIÓN */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="text-indigo-600" />
              <h2 className="text-xl font-bold">Realizar Predicción</h2>
            </div>

            {/* SLIDERS Y CHECKBOX */}
            <div className="space-y-4">
              <div>
                <label className="block mb-1 text-sm">Edad: {formData.edad}</label>
                <input
                  type="range"
                  min="20"
                  max="90"
                  value={formData.edad}
                  onChange={e => setFormData({ ...formData, edad: Number(e.target.value) })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block mb-1 text-sm">Años fumando: {formData.anos_fumando}</label>
                <input
                  type="range"
                  min="0"
                  max="60"
                  value={formData.anos_fumando}
                  onChange={e => setFormData({ ...formData, anos_fumando: Number(e.target.value) })}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  ['humo_segunda_mano', 'Humo segunda mano'],
                  ['historial_familiar', 'Historial familiar'],
                  ['exposicion_quimicos', 'Exposición química'],
                  ['enfermedad_pulmonar', 'Enfermedad pulmonar'],
                  ['tos_persistente', 'Tos persistente'],
                  ['dificultad_respirar', 'Dificultad al respirar'],
                  ['dolor_pecho', 'Dolor en el pecho']
                ].map(([key, label]) => (
                  <label key={key} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={formData[key] === 1}
                      onChange={e => setFormData({ ...formData, [key]: e.target.checked ? 1 : 0 })}
                    />
                    {label}
                  </label>
                ))}
              </div>

              <button
                onClick={predict}
                disabled={!model}
                className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-400"
              >
                Predecir Riesgo
              </button>

              {prediction && (
                <div className={`mt-4 p-4 rounded-lg border ${prediction.risk ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    {prediction.risk ? (
                      <AlertCircle className="text-red-600" size={24} />
                    ) : (
                      <CheckCircle className="text-green-600" size={24} />
                    )}
                    <h3 className="font-bold text-lg">
                      {prediction.risk ? 'RIESGO ALTO' : 'RIESGO BAJO'}
                    </h3>
                  </div>
                  <p>
                    Probabilidad estimada: <strong>{prediction.probability}%</strong>
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6 bg-white rounded-lg shadow p-6">
          <h3 className="font-bold mb-2">ℹ️ Información del Sistema</h3>
          <p>• Modelo: Regresión Logística Entrenada</p>
          <p>• Variables: 9 factores clínicos</p>
          <p>• Uso: herramienta educativa, no diagnóstico médico</p>
        </div>
      </div>
    </div>
  );
};

export default LungCancerPredictor;
