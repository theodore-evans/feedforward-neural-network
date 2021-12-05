#ifndef NEURALNET_H_INCLUDED
#define NEURALNET_H_INCLUDED

//#include <fstream>

namespace neuralnet
{
    enum LayerType {
        INPUT,
        BIAS,
        HIDDEN,
        OUTPUT,
        ERROR
    };

    class Neuron;
    class NeuronLayer;

    class NeuralNet
    {
        friend std::ostream & operator<<(std::ostream &os, const NeuralNet &net);

        protected: // inherited by NeuralNetIO
    std::   string dataFilename_;
    std::   string outputFilename_;
            size_t numInputNeurons_ ;
            size_t numHiddenLayers_ ;
            size_t numHiddenNeurons_;
            size_t numOutputNeurons_;
            size_t numTrainExamples_;
            size_t divergenceThreshold_;
            double learningRate_;
            double momentum_;
            double errorThreshold_;
            double dataScaleFactor_;

            size_t dataRow_;
            double dataBuffer_;

    std::   ifstream fetch_;
    std::   ofstream fout_;

    std::   vector<double>  inputData_;
    std::   vector<double>  trainingData_;

    std::   vector<NeuronLayer*> layers_;

        public:
            ~NeuralNet();
            NeuralNet(const config::Config &config);

            double* getInputPointer(const int index);  // ***
            double* getOutputPointer(const int index);

            bool initializeNetLinkage() const;

            void feedforwardNet()       const;
            void updateNetWeights()  const;

            bool loadData();
            bool updateInput();

            void trainNetwork();
            void runNetwork();

            NeuronLayer* operator[](const int layerIndex);
    };

    std::   ostream & operator<<(std::ostream &os, const NeuralNet &net);

    //////////////////////////////////////////////////

    class NeuronLayer
    {
        friend std::ostream & operator<<(std::ostream &os, const NeuronLayer &layer);

        private:
            NeuralNet* network_;
            int index_;
            LayerType layerType_;

    std::   vector<Neuron*> members_;
    std::   vector<Neuron*>::iterator membersIt_;

        public:
            ~NeuronLayer();
            NeuronLayer() {}
            NeuronLayer(NeuralNet* net, const int index, const LayerType layerType, const int numMembers);

            int         getIndex()   const;
            size_t      getSize()    const;
            LayerType   getType()    const;
            NeuralNet*  getNetwork() const;

            Neuron* operator[](const int neuronIndex) const;

            bool initializeLayerLinkage()   const;

            void feedforwardLayer();

            bool updateLayerWeights(const double learningRate, const double momentum)     const;
    };

    std::   ostream &operator<<(std::ostream &os, const NeuronLayer &layer);

    //////////////////////////////////////////////////

    class Neuron
    {
        friend std::ostream & operator<<(std::ostream &os, const Neuron &neuron);

        protected:
            int ID_, index_;
            neuralnet::NeuronLayer* layer_;
            double activation_, output_, error_;

        public:
            int generateID();
            int ID()    const;

            virtual ~Neuron() {}

            Neuron();
            Neuron(neuralnet::NeuronLayer* layer);

    virtual std::string type()      const = 0;

            int     layerIndex()    const;

    virtual double  activation()    const;
    virtual double  output()        const;
    virtual double  error()         const;

    virtual void    feedforward() {}

    virtual double  getWeight(const int index) {return 0;}
    virtual double  getDelta(const int index) {return 0;}

    virtual bool    updateWeights(const double learningRate, const double momentum);
    virtual void    generateLinks(const int n) {}

    };

    std::   ostream & operator<<(std::ostream &os, const Neuron &neuron);

    class NHidden : virtual public Neuron
    {
        protected:
    std::   vector<Neuron*> sources_;
    std::   vector<Neuron*> sinks_;
    std::   vector<double> weights_;
    std::   vector<double> deltas_;

        public:
            ~NHidden() {}

            NHidden();
            NHidden(neuralnet::NeuronLayer* layer);

   virtual  void generateLinks(const int n);
            void randomizeWeights();

    std::   string type()       const;

            void feedforward();

    virtual void updateActivation();
    virtual void updateOutput();
    virtual void updateError();

            double getWeight(const int index);
            double getDelta(const int index);

            bool updateWeights(const double learningRate, const double momentum);
    };

    class NOutput : public NHidden
    {
        private:
            double* data_;

        public:
            ~NOutput() {}

            NOutput() {}
            NOutput(neuralnet::NeuronLayer* layer, double* const data);

    std::   string type()  const;

            void feedforward();
            void updateError();

            // linear transfer functions
            void updateOutput();
            bool updateWeights(const double learningRate, const double momentum);

    };

    class NError : public NHidden     // passive nodes that set the activation bias
    {
        public:
            ~NError() {}

            NError() {}
            NError(neuralnet::NeuronLayer* layer) : Neuron(layer) {}

            void generateLinks(const int n); // sources only

    std::   string type()  const;

            void feedforward();
            void updateError();

    };

    class NInput : public Neuron
    {
        private:
            double* data_;

        public:
            ~NInput() {}

            NInput() {}
            NInput(neuralnet::NeuronLayer* layer, double* const data);

    std::   string type()   const;
            double output() const;

    };

    class NBias : public Neuron     // passive nodes that set the activation bias
    {
        private:

        public:
            ~NBias() {}

            NBias() {}
            NBias(neuralnet::NeuronLayer* layer) : Neuron(layer) {}

    std::   string type()   const;
            double output() const;
    };

    //////////////////////////////////////////////////

    double sigmoid(const double t, const double response); // Sigmoid response function, maybe include this is seperate header? EDIT: no.
    double sigmoidDerivative(const double t, const double response);
    double tanhDerivative(const double t);
}

#endif // NEURALNET_H_INCLUDED
