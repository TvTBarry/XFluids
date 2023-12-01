#pragma once

#include "global_setup.h"

// ================================================================================
// // // class AppendParas Member function definitions
// // // hiding implementation of template function causes undefined linking error
// ================================================================================
class AppendParas
{
private:
	int argc;
	char **argv;

public:
	AppendParas(){};
	~AppendParas(){};
	AppendParas(int argc, char **argv) : argc(argc), argv(argv){};

	/**
	 * string split
	 * @param str is the std::string that is to be splitted
	 * @param split is the separator to split str
	 * @return std::vector<T> output
	 */
	std::vector<std::string> Stringsplit(std::string str, const char split = ',')
	{
		std::string token;					 // recive buffers
		std::istringstream iss(str);		 // input stream
		std::vector<std::string> str_output; // recive buffers

		while (getline(iss, token, split)) // take "split" as separator
		{
			str_output.push_back(token);
		}

		return str_output;
	}

	std::vector<std::string> match(std::string option)
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < argc; i++)
		{
			if (std::string(argv[i]).find(option) != std::string::npos)
			{
				return Stringsplit(std::string(argv[i]).erase(0, option.length()));
				break;
			}
		}
		return std::vector<std::string>();
	};

	template <typename T>
	std::vector<T> Stringsplit(std::string str, const char split = ',')
	{
		bool error = false;
		std::string token;					 // recive buffers
		std::vector<T> output;				 // recive buffers
		std::istringstream iss(str);		 // input stream
		std::vector<std::string> str_output; // recive buffers

		while (getline(iss, token, split)) // take "split" as separator
		{
			str_output.push_back(token);
		}
		if (typeid(T) == typeid(int))
			std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
						   [](std::string &sr)
						   { return std::stoi(sr); });
		else if (typeid(T) == typeid(float))
			std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
						   [](std::string &sr)
						   { return std::stof(sr); });
		else if (typeid(T) == typeid(double))
			std::transform(str_output.begin(), str_output.end(), std::back_inserter(output),
						   [](std::string &sr)
						   { return std::stod(sr); });
		else
			std::cout << "Error: unsupportted return template type" << std::endl;

		return output;
	}

	template <typename T>
	std::vector<T> match(std::string option)
	{
		size_t i = 0;
		option += "=";
		for (i = 0; i < argc; i++)
		{
			if (std::string(argv[i]).find(option) != std::string::npos)
			{
				return Stringsplit<T>(std::string(argv[i]).erase(0, option.length()));
				break;
			}
		}
		return std::vector<T>();
	};
};
