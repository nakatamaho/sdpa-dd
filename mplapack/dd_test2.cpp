#include <iostream>
#include <random>
#include <qd/dd_real.h>

#define DD_FMA(a, b, c) __builtin_fma((a), (b), (c))
// ... (マクロ定義省略) ...

int main() {
    using namespace std;
    using namespace qd;

    cout.precision(33);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    dd_real da, db;
    da.x[0] = dist(mt);
    da.x[1] = dist(mt) * 1e-16;
    db.x[0] = dist(mt);
    db.x[1] = dist(mt) * 1e-16;

    dd_real sum = da + db;
    dd_real product = da * db;

    cout << "da = " << da << endl;
    cout << "db = " << db << endl;
    cout << "da + db = " << sum << endl;
    cout << "da * db = " << product << endl;

    cout << "macro test" << endl;
    DD_ADD(da, db, sum);
    cout << "da + db (macro) = " << sum << endl;

    return 0;
}
