#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "neuralnet.h"
#include "neuron.h"

//Neuron

    Neuron::Neuron() {}
    Neuron::Neuron(NeuronLayer* layer) : layer_(layer) {}

    int     Neuron::layerIndex()    const { return layer_->getIndex(); } // ?
    string  Neuron::type()          const { return "undefined"; }
    double  Neuron::activation()    const { return 0; }

    ostream & neuralnet::operator<<(ostream &os, const Neuron &rhs)
    {
        os  << "\t" << rhs.type() << " neuron in layer " << rhs.layerIndex()
            << "; Activation function: " << rhs.activation()
            << "; Output: "              << rhs.output()
            << endl;
            return os;
    }

//NActive (NHidden and NOutput)

    NActive::NActive() {}
    NActive::NActive(NeuronLayer* layer) : Neuron(layer)
    {
            generateSources();
            generateWeights();
    }

    void NActive::generateSources()
    {
        NeuralNet*   network( layer_->getNetwork() );                          // get a pointer to the neural network object
        NeuronLayer* sourceLayer = (*network)[layerIndex() - 1];// get the address of the previous laye

        for (size_t i(0); i < sourceLayer->getSize(); ++i) {
            sources_.push_back((*sourceLayer)[i]);
        }                               // populate vector of sources with all its members
    }

    void NActive::generateWeights()
    {
        for (size_t i(0); i < sources_.size(); ++i) {
            weights_.push_back( doubleRand() ); // random number between -1 and 1 (util.h)
        }
    }

    string NActive::type() const { return "Active"; }

    double NActive::activation() const
    {
        double weightedSum(0);
        for (size_t i(0); i < sources_.size(); ++i) {
            weightedSum += weights_[i] * sources_[i]->output(); // Sum over w(i)*x(i)
        }
        return weightedSum;
    }

    double NActive::output() const { return sigmoid( activation() ); }


//NInput

    string NInput::type()   const { return "Input"; }
    double NInput::output() const { return 1;       }


    //NBias

    string NBias::type()    const { return "Bias";  }
    double NBias::output()  const { return -1;      }


//NError

    string NError::type()   const { return "Error"; }
    double NError::output() const { return 0;       }

    //////////////////////////////////////////////////

double neuralnet::sigmoid(const double t)
{
    double response(0.1);
    return  1 / (1 + exp(-t / response) );
}
