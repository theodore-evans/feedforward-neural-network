#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include "neuralnet.h"

namespace neuron
{
    class Neuron
        {
            friend std::ostream & operator<<(std::ostream &os, const Neuron &neuron);

            protected:
                neuralnet::NeuronLayer* layer_;

            public:
                Neuron();
                Neuron(neuralnet::NeuronLayer* layer);

        virtual ~Neuron() {}

        virtual int layerIndex()    const;
        virtual std::string type()  const;

        virtual double activation() const;
        virtual double output()     const = 0;
        };

        std::   ostream & operator<<(std::ostream &os, const Neuron &neuron);

        class NActive : public Neuron
        {
            private:
        std::   vector<Neuron*> sources_;
        std::   vector<double> weights_;


            public:
                ~NActive() {}

                NActive();
                NActive(neuralnet::NeuronLayer* layer);

                void generateSources();
                void generateWeights();

        std::   string type()  const;

                double activation() const;
                double output()     const;

        };

        typedef NActive NHidden, NOutput;    // output and hidden neurons are the same thing

        class NInput : public Neuron
        {
            private:

            public:
                ~NInput() {}

                NInput() {}
                NInput(neuralnet::NeuronLayer* layer, const double* inputData) : Neuron(layer) {}

        std::   string type()  const;
                double output()     const;

        };

        class NBias : public Neuron     // passive nodes that set the activation bias
        {
            private:

            public:
                ~NBias() {}

                NBias() {}
                NBias(neuralnet::NeuronLayer* layer) : Neuron(layer) {}

        std::   string type()  const;
                double output()     const;

        };

        class NError : public Neuron     // passive nodes that set the activation bias
        {
            private:

            public:
                ~NError() {}

                NError() {}
                NError(neuralnet::NeuronLayer* layer, const double* outputData) : Neuron(layer) {}

        std::   string type()   const;
                double output() const;

        };

        //////////////////////////////////////////////////

        double sigmoid(const double t); // Sigmoid response function, maybe include this is seperate header? EDIT: no.
}

#endif // NEURON_H_INCLUDED
