#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>

#include "config.h"
#include "neuralnet.h"

using namespace std;
using namespace neuralnet;
using namespace config;


//NeuralNet
    NeuralNet::~NeuralNet()
    {
        for (vector<NeuronLayer*>::iterator vecIt = layers_.begin(); vecIt < layers_.end(); ++vecIt)
        {
            delete (*vecIt);
        }
        inputData_.clear();
        trainingData_.clear();
    }

    NeuralNet::NeuralNet(const Config &config) :

        // initialise parameters from static config variables
        dataFilename_    ( config.getStringParameter("dataFilename")),
        outputFilename_  ( config.getStringParameter("outputFilename")),
        numInputNeurons_ ( (size_t)config.getNumericalParameter("numInputNeurons")),
        numHiddenLayers_ ( (size_t)config.getNumericalParameter("numHiddenLayers")),
        numHiddenNeurons_( (size_t)config.getNumericalParameter("numHiddenNeurons")),
        numOutputNeurons_( (size_t)config.getNumericalParameter("numOutputNeurons")),
        numTrainExamples_( (size_t)config.getNumericalParameter("numTrainExamples")),
        divergenceThreshold_ ((size_t)config.getNumericalParameter("divergenceThreshold")),
        learningRate_    ( config.getNumericalParameter("learningRate")),
        momentum_        ( config.getNumericalParameter("momentum")),
        errorThreshold_  ( config.getNumericalParameter("errorThreshold")),
        dataScaleFactor_ ( config.getNumericalParameter("dataScaleFactor")),
        dataRow_(0)
    {
        // initialise associated input/output object
        inputData_.resize(numInputNeurons_);
        trainingData_.resize(numOutputNeurons_);

        cout << "Loading " << dataFilename_ << "... ";
        bool loadSuccess = loadData();
        if (!loadSuccess)
        {
            string errorOpenData("Could not open data file ");
            errorOpenData += dataFilename_;
            throw errorOpenData;
        }
        cout << "done" << endl;

        cout << "\nCreating network...\n";
        // NeuronLayer constructors take the arguments (network pointer, layer index, layer type, number of members)
        // Input layer
        cout << "\tGenerating input layer... ";
        layers_.push_back(new NeuronLayer(this, 0, INPUT, numInputNeurons_));
        cout << "done\n";

        // hidden layer(s)
        cout << "\tGenerating hidden layers... ";
        for (size_t i(0); i < numHiddenLayers_; ++i) {
            layers_.push_back(new NeuronLayer(this, 1 + i, HIDDEN, numHiddenNeurons_));
        }
        cout << "done\n";

        // output layer
        cout << "\tGenerating output layer... ";
        layers_.push_back(new NeuronLayer(this, numHiddenLayers_ + 1, OUTPUT, numOutputNeurons_));
        cout << "done\n";

        // error layer
        cout << "\tGenerating error layer... ";
        layers_.push_back(new NeuronLayer(this, numHiddenLayers_ + 2, ERROR, 1));
        cout << "done\n\n";

        cout << "Initializing synaptic connections...\n";
        bool linkingSuccess = initializeNetLinkage();
        if (!linkingSuccess) throw "Could not generate synaptic links.";
    }

    bool NeuralNet::initializeNetLinkage() const
    {
        if (layers_.empty()) return false;
        for (size_t i(0); i < layers_.size(); ++i)
        {
            cout << "\tLayer " << i << "... ";
            layers_[i]->initializeLayerLinkage();
            cout << "done\n";
        }
        return true;
    }

    bool NeuralNet::loadData()
    {
        fetch_.open(dataFilename_.c_str());

        if (!fetch_) return false;

        while (fetch_.peek() == '#') fetch_.ignore(256, '\n'); // ignore comments
        return true;
    }

    bool NeuralNet::updateInput() // add error checking!
    {
        ++dataRow_;
        for (size_t i(0); i < numInputNeurons_; ++i)
        {
            fetch_ >> dataBuffer_;
            inputData_[i] = dataBuffer_;
        }

        for (size_t i(0); i < numOutputNeurons_; ++i)
        {
            fetch_ >> dataBuffer_;
            trainingData_[i] = dataScaleFactor_ * dataBuffer_;
        }

        if (!fetch_.good()) return false;
        return true;
    }

    void NeuralNet::feedforwardNet() const
    {
        for (size_t i(0); i < layers_.size(); ++i) {
            layers_[i]->feedforwardLayer();
        }
    }

    void NeuralNet::updateNetWeights() const
    {
        for (size_t i(layers_.size() - 1); i > 1; --i) {
            layers_[i]->updateLayerWeights(learningRate_, momentum_);
        }
    }

    double* NeuralNet::getInputPointer(const int index)  { return &inputData_[index];  }
    double* NeuralNet::getOutputPointer(const int index) { return &trainingData_[index]; }

    void NeuralNet::trainNetwork()
    {
        NeuronLayer* errorLayer = layers_[numHiddenLayers_ + 2];
        NeuronLayer* outputLayer = layers_[numHiddenLayers_ + 1];

        Neuron* errorNode = (*errorLayer)[0];

        fout_.open(outputFilename_.c_str());
        if (!fout_.good()) throw "Could not open output file.";

        size_t iteration(0);
        double error(0);
        // 1-off calculations
        double squareScaleFactor = dataScaleFactor_ * dataScaleFactor_;
        double reciprocalScaleFactor = 1 / dataScaleFactor_;
        double recipSquareScaleFactor = 1 / squareScaleFactor;
        double scaledErrorThreshold = squareScaleFactor * errorThreshold_;

        for (size_t n(0); n < numTrainExamples_; ++n)
        {
            updateInput();
            cout << "Training example " << dataRow_ << " of " << numTrainExamples_ << "... ";
            feedforwardNet();

            error = errorNode->error();

            for (size_t i(0); i < numOutputNeurons_; ++i)
            {
                fout_   << (*outputLayer)[i]->output() * reciprocalScaleFactor << " ";
            }
            fout_   << error * recipSquareScaleFactor << endl;

            iteration = 0;
            while (error > scaledErrorThreshold)
            {
                ++iteration;
                updateNetWeights();
                feedforwardNet();
                error = errorNode->error();
                if (iteration >= divergenceThreshold_) throw "Convergence failure. Try reducing learning rate.";
            }
            cout << "done in " << iteration << " steps.\n";
        }

        while (updateInput())   //
        {
            for (size_t i(0); i < numOutputNeurons_; ++i)
            {
                fout_   << (*outputLayer)[i]->output() * reciprocalScaleFactor << " ";
            }
            fout_   << error * recipSquareScaleFactor << endl;

            feedforwardNet();
            error = errorNode->error();
        }
    }

    // accessor for layers* in net
    NeuronLayer* NeuralNet::operator[](const int layerIndex) { return layers_[layerIndex]; }

    ostream & neuralnet::operator<<(ostream &os, const NeuralNet &rhs)
    {
        for (size_t i(0); i < rhs.layers_.size(); ++i) {
            os << *rhs.layers_[i];
        }
        return os;
    }

    //////////////////////////////////////////////////

//NeuronLayer

    NeuronLayer::~NeuronLayer()
    {
        for (vector<Neuron*>::iterator vecIt = members_.begin(); vecIt < members_.end(); ++vecIt) {
            delete (*vecIt);
        }
        members_.clear();
    }

    // parameterised constructor
    NeuronLayer::NeuronLayer(NeuralNet* network, const int index, const LayerType layerType, const int numMembers)
                   : network_(network), index_(index), layerType_(layerType)
    {
        switch (layerType_)
        {

            case INPUT: // create an input layer of N input neurons + 1 bias neuron
                for (int i(0); i < numMembers; ++i) {
                    members_.push_back(new NInput(this, network->getInputPointer(i) ));//, data[i]));
                }
                members_.push_back(new NBias(this));
                break;

            case HIDDEN: // create a hidden layer of N hidden neurons + 1 bias neuron
                for (int i(0); i < numMembers; ++i) {
                    members_.push_back(new NHidden(this));
                }
                members_.push_back(new NBias(this));
                break;

            case OUTPUT: // create an output layer of N output neurons only
                for (int i(0); i < numMembers; ++i) {
                    members_.push_back(new NOutput(this, network->getOutputPointer(i)));
                }
                break;

            case ERROR:
                for (int i(0); i < numMembers; ++i) {
                    members_.push_back(new NError(this));
                }
                break;

            default:
                throw "Bad node type.";
                break;
        }

    }

    int         NeuronLayer::getIndex()   const { return index_;          }
    size_t      NeuronLayer::getSize()    const { return members_.size(); }
    LayerType   NeuronLayer::getType()    const { return layerType_;      }
    NeuralNet*  NeuronLayer::getNetwork() const { return network_;        }

    Neuron*     NeuronLayer::operator[](const int neuronIndex) const
    {
        return members_[neuronIndex];
    }

    bool NeuronLayer::initializeLayerLinkage() const
    {
        if (members_.empty()) return false;
        for (size_t i(0); i < members_.size(); ++i) {
            members_[i]->generateLinks(i);
        }
        return true;
    }

    void NeuronLayer::feedforwardLayer()
    {
        for (size_t i(0); i < members_.size(); ++i) {
            members_[i]->feedforward();

        }
    }
    bool NeuronLayer::updateLayerWeights(const double learningRate, const double momentum) const
    {
        if (members_.empty()) return false;
        for (size_t i(0); i < members_.size(); ++i) {
            members_[i]->updateWeights(learningRate, momentum);
        }
        return true;
    }

    ostream & neuralnet::operator<<(ostream &os, const NeuronLayer &rhs)
    {
        os << setprecision(3) << fixed;
        os << "\nNeuron layer " << rhs.index_ << ", size " << rhs.members_.size();

        os << "\nMembers:\n";
        for (size_t i(0); i < rhs.members_.size(); ++i) {
            cout << i << " ";
            os << *rhs.members_[i];
        }
        return os;
    }

    //////////////////////////////////////////////////

//Neuron ABC

    Neuron::Neuron() {}
    Neuron::Neuron(NeuronLayer* layer) : ID_(generateID()), index_(0), layer_(layer),
                                        activation_(0), output_(0), error_(0)
    {}

    int Neuron::generateID() // DEBUGGING ONLY
    {
        static int staticID;
        return staticID++;
    }

    int     Neuron::ID()         const { return ID_; } // DEBUGGING ONLY
    int     Neuron::layerIndex() const { return layer_->getIndex(); } // ?

    double  Neuron::activation() const { return activation_; }
    double  Neuron::output()     const { return output_;     }
    double  Neuron::error()      const { return error_;      }

    bool    Neuron::updateWeights(const double learningRate, const double momentum) {return true;}

    // DEBUGGING ONLY
    ostream & neuralnet::operator<<(ostream &os, const Neuron &rhs)
    {
        os << "[" << rhs.ID() << "] "    << rhs.type() << " (" << rhs.layerIndex() << ")"
            << "; Activation: "          << rhs.activation()
            << "; Output: "              << rhs.output()
            << "; Error: "               << rhs.error()
            << endl;
            return os;
    }

//NHidden

    NHidden::NHidden() {}
    NHidden::NHidden(NeuronLayer* layer) : Neuron(layer)
    {
    }

    void NHidden::generateLinks(const int n) // populate vectors with pointers to anterior & posterior nodes
    {
        index_ = n;
        NeuralNet*   network     = (*layer_).getNetwork();        // get a pointer to the neural network object

        NeuronLayer* sourceLayer = (*network)[layerIndex() - 1];
        sources_.resize(sourceLayer->getSize());

        for (size_t i(0); i < sources_.size(); ++i)
        {
            sources_[i] = (*sourceLayer)[i];            // assign link to source nodes
        }

        NeuronLayer* sinkLayer   = (*network)[layerIndex() + 1];
        sinks_.resize(sinkLayer->getSize());

        for (size_t i(0); i < sinks_.size(); ++i)
        {
            sinks_[i] = (*sinkLayer)[i];                // assign link to sink node
        }

        randomizeWeights();
    }

    void NHidden::randomizeWeights()
    {
        weights_.resize(sources_.size());
        deltas_.resize(sources_.size());

        for (size_t i(0); i < sources_.size(); ++i) {
            weights_[i] = 2 * ( (double)rand() / ( (double)RAND_MAX + 1.0) ) - 1.0; // random number between -1 and 1
            deltas_[i] = 0;
        }
    }

    string NHidden::type()      const { return "Hidden"; }

    void NHidden::feedforward()
    {
        updateActivation();
        updateOutput();
    }

    void NHidden::updateActivation()
    {
        double weightedSum(0);
        for (size_t i(0); i < sources_.size(); ++i) {
            weightedSum += weights_[i] * sources_[i]->output(); // Sum over w(i)*x(i)
        }
        activation_ = weightedSum;
    }

    void NHidden::updateOutput()
    {
        output_ = sigmoid(activation(), 1);
    }

    void NHidden::updateError()
    {
        double error(0);
        for (size_t i(0); i < sinks_.size(); ++i)
        {
            error += sinks_[i]->getDelta(index_) * sinks_[i]->getWeight(index_);
        }
        error_ = error;
    }

    double NHidden::getWeight(const int index) { return weights_[index]; }
    double NHidden::getDelta(const int index) { return deltas_[index]; }

    bool NHidden::updateWeights(const double learningRate, const double momentum)
    {
        updateError();
        for (size_t i(0); i < sources_.size(); ++i)
        {
            double previousDelta = deltas_[i];
            deltas_[i] = learningRate * error() * sigmoidDerivative(activation(), 1) * sources_[i]->output();
            weights_[i] += deltas_[i] + momentum * previousDelta;
        }
        return true;
    }

//NOutput

    NOutput::NOutput(NeuronLayer* layer, double* const data)
            : Neuron(layer), data_(data)
    {
    }

    string NOutput::type()   const { return "Output"; }

    void NOutput::feedforward()
    {
        updateActivation();
        updateOutput();
        updateError();
    }

    void NOutput::updateError()   // error function, (t - o)
    {
        error_ = (*data_ - output_);
    }

    // Linear transfer functions
    void NOutput::updateOutput()
    {
        output_ = activation();
    }

    bool NOutput::updateWeights(const double learningRate, const double momentum)
    {
        updateError();
        for (size_t i(0); i < sources_.size(); ++i)
        {
            double previousDelta = deltas_[i];
            deltas_[i] = learningRate * error() * sources_[i]->output();
            weights_[i] += deltas_[i] + momentum * previousDelta;
        }
        return true;
    }

//NError

    void NError::generateLinks(const int n)
    {
        index_ = n;

        NeuralNet*   network     = (*layer_).getNetwork();        // get a pointer to the neural network object
        NeuronLayer* sourceLayer = (*network)[layerIndex() - 1];  // get the address of the previous

        sources_.resize(sourceLayer->getSize());
        for (size_t i(0); i < sources_.size(); ++i) {
            sources_[i] = (*sourceLayer)[i];
        }

        randomizeWeights();
    }

    string NError::type()   const { return "Error"; }

    void NError::feedforward()
    {
        updateError();
    }

    void NError::updateError()
    {
        double error(0);
        double totalError(0);

        for (size_t i(0); i < NHidden::sources_.size(); ++i)
        {
            error = sources_[i]->error();
            totalError += error * error / 2; // Sum[ (t - o)^2 /2 ]
        }
        error_ = totalError;
    }

//NInput
    NInput::NInput(neuralnet::NeuronLayer* layer, double* const data) : Neuron(layer), data_(data)
    {}

    string NInput::type()   const { return "Input"; }
    double NInput::output() const { return *data_; }

//NBias

    string NBias::type()    const { return "Bias";  }
    double NBias::output()  const { return -1;      }

    //////////////////////////////////////////////////

double neuralnet::sigmoid(const double t, const double response)
{
    return  1 / (1 + exp(-t / response) );
}

double neuralnet::sigmoidDerivative(const double t, const double response)
{
    double exponent = exp(t/response);
    return exponent/(response * (1 + exponent) * (1 + exponent));
}


