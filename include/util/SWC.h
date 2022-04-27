#ifndef KOUEK_SWC_H
#define KOUEK_SWC_H

#include <set>
#include <fstream>

namespace kouek
{
    // CNIC's Structure Identifier in SWC format
    enum class SWCNodeType : int
    {
        Undefined = 0,
        Soma,           // Soma 1
        Axon,           // Axon 2
        Dendrite,       // (basal) dendrite 3
        ApicalDendrite, // apical dendrite 4
        ForkPoint,      // fork point 5
        EndPoint,       // end point 6
        Custom
    };

    struct SWCNode
    {
        int64_t id, parentID;
        SWCNodeType type;
        double x, y, z;
        double radius;
    };

    class SWC
    {
    protected:
        std::vector<std::string> commentLines;

        struct LessSWCNode
        {
            bool operator()(const SWCNode& a, const SWCNode& b) const
            {
                return a.id < b.id;
            }
        };
        std::set<SWCNode, LessSWCNode> nodes;

    public:
        virtual void add(const SWCNode& node) = 0;
        inline void clear()
        {
            nodes.clear();
        }
        inline const auto& getNodes() const
        {
            return nodes;
        }
    };

    class FileSWC : public SWC
    {
    private:
        std::string filePath;
        char* itr;
        long line;

    private:
        inline void next()
        {
            itr++;
        }

        inline bool accept(char ch)
        {
            if (*itr == ch)
            {
                itr++;
                return true;
            }
            return false;
        }

        inline bool acceptWhiteSpace()
        {
            if (accept(' ')) return true;
            if (accept('\t')) return true;
            return false;
        }

        inline bool acceptEndOfLine()
        {
            accept('\r'); // allow \r\n
            if (accept('\n')) return true;
        }

        bool parseLine()
        {
            using namespace std;
            // allow multiple white spaces at head of line
            while (acceptWhiteSpace())
                ;
            // allow space line, but don't accept
            if (*itr == '\0') return false;
            if (acceptEndOfLine()) return true;
            // record all comments(comments shoud be placed at head-lines of file)
            if (accept('#'))
            {
                const char* commentStart = itr - 1; // start from #
                while (!acceptEndOfLine() && *itr != '\0')
                    next();
                const char* commentEnd = itr;
                commentLines.push_back(string(commentStart, commentEnd));
                return true;
            }
            // deal with 7 data terms per line
            SWCNode node;
            for (int i = 0; i < 7; i++)
            {
                char* intStrEnd = nullptr;
                char* doubleStrEnd = nullptr;
                // if it can be converted to Integer,
                //   intStrEnd will point to the first character after the number.
                long long intVal = strtoll(itr, &intStrEnd, 0);
                // if it can be converted to Float,
                //   intStrEnd will point to the first character after the number.
                double doubleVal = strtod(itr, &doubleStrEnd);

                if (intStrEnd <= itr && doubleStrEnd <= itr)
                    throw runtime_error("Data term cannot parsed to Integer or Float(Double), at " +
                        to_string((size_t)itr));
                switch (i)
                {
                case 0: // Sample Number or id: Integer
                case 1: // Structure Identifier or type: Integer
                    if (intStrEnd <= itr)
                        throw runtime_error("Data term 0 or 1 must be Integer, at " + to_string((size_t)itr));
                    itr = intStrEnd;
                    break;
                case 2: // x: Double
                case 3: // y: Double
                case 4: // z: Double
                case 5: // Radius: Double
                    if (doubleStrEnd <= itr)
                        throw runtime_error("Data term 2 to 5 must be Double, at " + to_string((size_t)itr));
                    itr = doubleStrEnd;
                    break;
                case 6: // Parent's Structure Identifier: Integer
                    if (intStrEnd <= itr) throw runtime_error("Data term 6 must be Integer, at " + to_string((size_t)itr));
                    itr = intStrEnd;
                    break;
                }

                switch (i)
                {
                case 0: // Sample Number or id: Integer
                    node.id = intVal;
                    break;
                case 1: // Structure Identifier or type: Integer
                    node.type = static_cast<SWCNodeType>(intVal);
                    break;
                case 2: // x: Double
                    node.x = doubleVal;
                    break;
                case 3: // y: Double
                    node.y = doubleVal;
                    break;
                case 4: // z: Double
                    node.z = doubleVal;
                    break;
                case 5: // Radius: Double
                    node.radius = doubleVal;
                    break;
                case 6: // Parent's Structure Identifier: Integer
                    node.parentID = intVal;
                    break;
                }
            }
            nodes.insert(node);
            // allow multiple white spaces at tail of line
            while (acceptWhiteSpace())
                ;
            if (acceptEndOfLine()) return true;
            return false;
        }

    public:
        FileSWC(std::string_view swcPath) : filePath(swcPath), itr(nullptr)
        {
            using namespace std;

            ifstream in(filePath.data(), ios::ate | ifstream::binary);
            if (!in.is_open()) throw runtime_error("Cannot open file: " + filePath);

            auto fileSize = in.tellg();
            in.seekg(ios::beg);

            char* buffer = new char[static_cast<size_t>(fileSize) + 1];
            in.read(buffer, fileSize);
            buffer[static_cast<size_t>(fileSize)] = '\0';

            in.close();

            try
            {
                itr = buffer;
                while (parseLine())
                    line++;
                if (*itr != '\0')
                    throw runtime_error("Unexpected syntax at line: " + to_string(line) +
                        ", syntax pos from filehead: " + to_string((size_t)itr));
            }
            catch (exception& e)
            {
                throw runtime_error(string("Parse SWC file FAILED: ") + e.what());
            }

            delete[] buffer;
        }
        ~FileSWC()
        {
            using namespace std;
            ofstream out(filePath, ios::out);
            if (!out.is_open()) throw runtime_error("Cannot open file: " + filePath);

            size_t cnt = 1;
            for (auto& node : nodes)
            {
                out << to_string(node.id) << ' ' << to_string(static_cast<int>(node.type)) << ' ' << to_string(node.x)
                    << ' ' << to_string(node.y) << ' ' << to_string(node.z) << ' ' << to_string(node.radius) << ' '
                    << to_string(node.parentID);
                if (cnt != nodes.size()) out << endl;
                cnt++;
            }
        }
        void add(const SWCNode& node) override
        {
            nodes.insert(node);
        }
    };
}

#endif // !KOUEK_SWC_H
