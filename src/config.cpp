#include <iostream>
#include <fstream>
#include <string>
//#include <map>    // included in config.h

#include "config.h"

using namespace std;
using namespace config;

Config::~Config() {
    nParameters_.clear();   // Clear map objects
    sParameters_.clear();   //
}

Config::Config(const char* filename) : filename_(filename)
{
    bool loadSuccess = loadParameters();
    if (!loadSuccess)
    {
        string errorOpenConfig("Could not open configuration file ");
        errorOpenConfig += filename;
        throw errorOpenConfig;
    }

    cout << filename << " loaded." << endl;
}

bool Config::loadParameters()
{
    ifstream fetch(filename_);

    if (!fetch) return false;

    string parameterType(""), parameterName(""), sParameterValue("");    // new empty key and value objects
    double nParameterValue(0);

    while ( fetch.good() )
    {
        fetch >> parameterType;     // each line has format:
        fetch >> parameterName;     // [type] [name] [value]

        if  (parameterType  == "int" || parameterType == "double")
        {
            fetch >> nParameterValue;
            nParameters_[parameterName] = nParameterValue;
        }

        else if (parameterType  == "string")
        {
            fetch >> sParameterValue;
            sParameters_[parameterName] = sParameterValue;
        }

        else if (parameterType == "//") {
            fetch.ignore( 256, '\n');} // ignore comments

        else
        {   // throw an error for improper type or bad formatting
            string errorBadType(parameterName);
            errorBadType += "has invalid type \'";
            errorBadType += parameterType;
            errorBadType += "\'";
            throw errorBadType;
        }
    }

    return true;
}

double Config::getNumericalParameter(string parameterName) const
{
    if  ( nParameters_.find(parameterName) == nParameters_.end() )
    {
        string errorBadParameter("Invalid parameter \'");
        errorBadParameter += parameterName;
        errorBadParameter += "\'";
        throw errorBadParameter;
    }

    else return nParameters_.find(parameterName)->second;
}

string Config::getStringParameter(string parameterName) const
{
    if  ( sParameters_.find(parameterName) == sParameters_.end() )
    {
        string errorBadParameter("Invalid parameter \'");
        errorBadParameter += parameterName;
        errorBadParameter += "\'";
        throw errorBadParameter;
    }

    else return sParameters_.find(parameterName)->second;
}

std::ostream & config::operator<<(std::ostream &os, const Config &rhs)
{
    for (sMap_t::const_iterator sMapIt = rhs.sParameters_.begin(); sMapIt != rhs.sParameters_.end(); ++sMapIt) {
        os << sMapIt->first << ":  '" << sMapIt->second << "'" << endl;
    }

    for (nMap_t::const_iterator nMapIt = rhs.nParameters_.begin(); nMapIt != rhs.nParameters_.end(); ++nMapIt) {
        os << nMapIt->first << ":  " << nMapIt->second << endl;
    }

    return os;
}
