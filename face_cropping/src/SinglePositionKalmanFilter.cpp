#include "SinglePositionKalmanFilter.h"

SinglePositionKalmanFilter::SinglePositionKalmanFilter(
    float initialPosition,
    float initialVelocity,
    float initialPositionVariance,
    float initialVelocityVariance,
    float qPosition,
    float qVelocity)
    : m_muX(initialPosition),
      m_muV(initialVelocity),
      m_sigmaXX(initialPositionVariance),
      m_sigmaVV(initialVelocityVariance),
      m_qX(qPosition),
      m_qV(qVelocity)
{
}

void SinglePositionKalmanFilter::update(float newPosition, float rPosition, float dt)
{
    float muTX = m_muX;
    float muTV = m_muV;

    float sigmaTXX = m_sigmaXX;
    float sigmaTVV = m_sigmaVV;

    // Prediction
    float muPX = muTX + muTV * dt;
    float muPV = muTV;

    float dtSquared = dt * dt;
    float sigmaPXX = sigmaTXX + sigmaTVV * dtSquared + m_qX * dt;
    float sigmaPVV = sigmaTVV + m_qV * dt;
    float sigmaPXV = sigmaTVV * dt;

    // Update
    m_muX = muPX + sigmaPXX / (sigmaPXX + rPosition) * (newPosition - muPX);
    m_muV = muPV + sigmaPXV / (sigmaPXX + rPosition) * (newPosition - muPX);

    float rPositionPrim = rPosition / (sigmaPXX + rPosition);
    float sigmaPXVSquared = sigmaPXV * sigmaPXV;
    m_sigmaXX = sigmaPXX * rPositionPrim;
    m_sigmaVV = sigmaPVV - sigmaPXVSquared / (sigmaPXX + rPosition);
}
