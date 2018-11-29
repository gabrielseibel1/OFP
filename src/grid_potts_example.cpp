#include "grid_potts_example.h"
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

#define LABELS 5

using namespace std; // 'using' is used only in example code
using namespace opengm;

// this function maps a node (x, y) in the grid to a unique variable index
inline size_t variableIndex(const size_t x, const size_t y, size_t nx) {
    return x + nx * y;
}

cv::Point labelDisplacement(int l) {
    switch (l) {
        default:
        case 0:
            return cv::Point(-1, -1); //upper left
        case 1:
            return cv::Point(-1, 0); //upper mid
        case 2:
            return cv::Point(-1, +1); //upper right
        case 3:
            return cv::Point(0, -1); //mid right
        case 4:
            return cv::Point(0, 0); //mid mid
        case 5:
            return cv::Point(0, +1); //mid right
        case 6:
            return cv::Point(+1, -1); //bot left
        case 7:
            return cv::Point(+1, 0); //bot mid
        case 8:
            return cv::Point(+1, +1); //bot right
    }
}

int costFunction(int width, int height, cv::Mat *img1, cv::Mat *img2, int x, int y, int l) {
    if (img1->at<int>(y, x) == 0) { //not an edge!
        return 0;
    }

    //normalized cross correlation
    int cc = 0;
    int radius = 3;
    int pointsCounted = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            //points are (x,y) and ((x,y) + V(x,y))
            cv::Point point1(y + dy, x + dx);
            cv::Point point2 = point1 + labelDisplacement(l);

            //if points are in matrix
            if (point1.x >= 0 && point1.y >= 0 && point2.x >= 0 && point2.y >= 0 &&
                point1.x < width && point1.y < height && point2.x < width && point2.y < height) {

                int pix1 = img1->at<int>(point1) / 255; //normalized (?)
                int pix2 = img2->at<int>(point2) / 255; //normalized (?)

                cc += pix1 * pix2;
                pointsCounted++;
            }
        }
    }
    int ncc = 0;
    if (pointsCounted > 0) ncc = cc / pointsCounted;
    return ncc;
}

void grid_potts_example(int width, int height, cv::Mat *img1, cv::Mat *img2) {
    auto nx = static_cast<size_t>(width);
    auto ny = static_cast<size_t>(height);
    double lambda = 0.1; // coupling strength of the Potts model

    // construct a label space with
    // - nx * ny variables
    // - each having numberOfLabels many labels
    typedef SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(nx * ny, LABELS);

    // construct a graphical model with
    // - addition as the operation (template parameter Adder)
    // - support for Potts functions (template parameter PottsFunction<double>)
    typedef GraphicalModel<int, Adder, OPENGM_TYPELIST_2(ExplicitFunction<int>, PottsFunction<int>), Space> Model;
    Model gm(space);

    // for each node (x, y) in the grid, i.e. for each variable
    // variableIndex(x, y) of the model, add one 1st order functions
    // and one 1st order factor
    for (size_t y = 0; y < ny; ++y)
        for (size_t x = 0; x < nx; ++x) {
            // function
            const size_t shape[] = {LABELS};
            ExplicitFunction<int> f(shape, shape + 1);
            for (size_t s = 0; s < LABELS; ++s) {
                f(s) = costFunction(width, height, img1, img2, x, y, s);
            }
            Model::FunctionIdentifier fid = gm.addFunction(f);

            // factor
            size_t variableIndices[] = {variableIndex(x, y, nx)};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

    // add one (!) 2nd order Potts function
    PottsFunction<int> f(LABELS, LABELS, 0.0, lambda);
    Model::FunctionIdentifier fid = gm.addFunction(f);

    // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
    // add one factor that connects the corresponding variable indices and
    // refers to the Potts function
    for (size_t y = 0; y < ny; ++y)
        for (size_t x = 0; x < nx; ++x) {
            if (x + 1 < nx) { // (x, y) -- (x + 1, y)
                size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x + 1, y, nx)};
                sort(variableIndices, variableIndices + 2);
                gm.addFactor(fid, variableIndices, variableIndices + 2);
            }
            if (y + 1 < ny) { // (x, y) -- (x, y + 1)
                size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x, y + 1, nx)};
                sort(variableIndices, variableIndices + 2);
                gm.addFactor(fid, variableIndices, variableIndices + 2);
            }
        }

    // set up the optimizer (loopy belief propagation)
    typedef BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
    typedef MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
    const size_t maxNumberOfIterations = 40;
    const double convergenceBound = 1e-7;
    const double damping = 0.5;
    BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
    BeliefPropagation bp(gm, parameter);

    // optimize (approximately)
    BeliefPropagation::VerboseVisitorType visitor;
    bp.infer(visitor);

    // obtain the (approximate) argmin
    vector<size_t> labeling(nx * ny);
    bp.arg(labeling);

    // output the (approximate) argmin
    size_t variableIndex = 0;
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            cout << labeling[variableIndex] << ' ';
            ++variableIndex;
        }
        cout << endl;
    }
}
