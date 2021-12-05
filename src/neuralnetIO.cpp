#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "neuralnet.h"
#include "neuralnetIO.h"

using namespace std;
using namespace neuralnet;
using namespace neuralnetio;

NeuralNetIO::~NeuralNetIO() {}
NeuralNetIO::NeuralNetIO() {}

NeuralNetIO::NeuralNetIO(NeuralNet* network) :
        network_(network),
        numInputNeurons_(network->numInputNeurons_),    // initialize local copies
        numOutputNeurons_(network->numOutputNeurons_),  // for sake of update function
{
    bool loadSuccess = loadData();
    if (!loadSuccess) {
        // throw poop
    }

    // initialise input/output data vectors and pointer vectors
    network_->inputData_.resize(numInputNeurons_);
    network_->inputData_.resize(numOutputNeurons_);


bool NeuralNetIO::loadData()
{
    fetch_.open(network->dataFilename_);

    if (!fetch_) return false; //throw your poo

    while (fetch_.peek() == '#') fetch_.ignore(256, '\n'); // ignore comments
    return true;
}

bool NeuralNetIO::updateData() // add error checking!
{
    for (size_t i(0); i < numInputNeurons_; ++i) // use local variables for speed
    {
        fetch_ >> inputData_[i];
    }

    for (size_t j(0); i < numOutputNeurons_; ++j)
    {
        fetch_ >> outputData_[j];
    }
}
