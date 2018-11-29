#include "grid_potts_example.h"
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

#define RADIUS 4
#define LABELS 81

using namespace std; // 'using' is used only in example code
using namespace opengm;

// this function maps a node (x, y) in the grid to a unique variable index
inline size_t variableIndex(const size_t x, const size_t y, size_t nx) {
    return x + nx * y;
}

unsigned int costFunction(int width, int height, unsigned char *img1, unsigned char *img2, int x, int y, int l) {

    //TODO change this to ncc + smoothness?


    int sum = 0;
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            int lx = l / (2 * RADIUS + 1) - RADIUS;
            int ly = l % (2 * RADIUS + 1) - RADIUS;
            int refPixel = img1[(y + dy) * width + x + dx];
            int tarPixel = img2[(y + dy) * width + x + dx - l];
            sum += abs(refPixel - tarPixel);
        }
    }
    auto avg = static_cast<unsigned int>(sum / LABELS);
    return avg;
}

void grid_potts_example(int width, int height, cv::Mat img1, cv::Mat img2) {
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
    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> , PottsFunction<double> ) , Space> Model;
    Model gm(space);

    // for each node (x, y) in the grid, i.e. for each variable
    // variableIndex(x, y) of the model, add one 1st order functions
    // and one 1st order factor
    for(size_t y = 0; y < ny; ++y)
        for(size_t x = 0; x < nx; ++x) {
            // function
            const size_t shape[] = {LABELS};
            ExplicitFunction<double> f(shape, shape + 1);
            for(size_t s = 0; s < LABELS; ++s) {
                f(s) = costFunction(/*TODO pass parameters*/);
            }
            Model::FunctionIdentifier fid = gm.addFunction(f);

            // factor
            size_t variableIndices[] = {variableIndex(x, y, nx)};
            gm.addFactor(fid, variableIndices, variableIndices + 1);
        }

    // add one (!) 2nd order Potts function
    PottsFunction<double> f(LABELS, LABELS, 0.0, lambda);
    Model::FunctionIdentifier fid = gm.addFunction(f);

    // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
    // add one factor that connects the corresponding variable indices and
    // refers to the Potts function
    for(size_t y = 0; y < ny; ++y)
        for(size_t x = 0; x < nx; ++x) {
            if(x + 1 < nx) { // (x, y) -- (x + 1, y)
                size_t variableIndices[] = {variableIndex(x, y, nx), variableIndex(x + 1, y, nx)};
                sort(variableIndices, variableIndices + 2);
                gm.addFactor(fid, variableIndices, variableIndices + 2);
            }
            if(y + 1 < ny) { // (x, y) -- (x, y + 1)
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
    for(size_t y = 0; y < ny; ++y) {
        for(size_t x = 0; x < nx; ++x) {
            cout << labeling[variableIndex] << ' ';
            ++variableIndex;
        }
        cout << endl;
    }
}
