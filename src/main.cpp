#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>
#include <ctime>

#include "config.h"
#include "neuralnet.h"

using namespace std;
using namespace neuralnet;
using namespace config;

int main()
{
    srand((unsigned)time(0));   // time seed for random number generator

    cout << "Persephone v0.1 2012: An artificial neural network, by Theo Evans" << endl;
    cout << "Enter configuration file name:" << endl << ">";
    char configFilename[64];
    cin >> configFilename;

    try
    {
        Config config(configFilename);

        string selection;
        bool finished(false);

        while(!finished)
        {
            selection.clear();
            cout << "\n[C]reate neural network\n[P]rint configuration\n[Q]uit\n>";
            cin >> selection;

            switch ( toupper(selection[0]) ) // get the first letter of input
            {
                case 'C':
                {
                    NeuralNet myNetwork(config);
                    cout << "\nNetwork created, press any key to begin training.\n";
                    cin.get();
                    cin.ignore();

                    myNetwork.trainNetwork();

                    finished = true;
                    break;
                }

                case 'P':
                {
                    cout << config;
                    break;
                }

                case 'Q':
                {
                    finished = true;
                    break;
                }

                default:
                {
                    cout << "Invalid selection." << endl;
                    break;
                }
            }
        }

        cout << "Press any key to exit.\n";
        cin.get();
    }
    catch(const string &err)
    {
        cout << "\nException: " << err << endl;
    }
    catch(exception &ex)
    {
        cout << "\nException: " << ex.what() << endl;
    }

    return 0;
}
