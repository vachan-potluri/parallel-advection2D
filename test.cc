/**
 * @brief For testing
 */

#include <iostream>

int main()
{
        using namespace std;
        int x=3;
        cout << "x before lambda " << x << endl;
        auto lambda = [&](){x++;};
        lambda();
        cout << "x after lambda " << x << endl;
}