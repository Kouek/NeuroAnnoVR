#ifndef KOUEK_SWC_CONVERTER_H
#define KOUEK_SWC_CONVERTER_H

#include <stack>
#include <unordered_map>
#include <unordered_set>

#include <util/SWC.h>
#include <renderer/GLPathRenderer.h>

namespace kouek
{
	class SWCConverter
	{
	private:
		using SWCIDTy = decltype(SWCNode::id);

	public:
		inline static std::vector<glm::vec3> colorTabl = {
			{1.f,.0f,.1f}, {1.f,1.f,.0f},
			{0.f,1.f,.5f}, {0.f,.5f,1.f}
		};

		static void appendSWCToGLPathRenderer(const SWC& swc, GLPathRenderer& pathRenderer)
		{
			// swcID -> swcIDs of its children
			std::unordered_map<SWCIDTy, std::vector<SWCIDTy>> tree;
			// swcID -> id in GLPathRenderer
			std::unordered_map<SWCIDTy, GLuint> idTabl;
			
			size_t colTablIdx = 0;
			auto increaseColTablIdx = [&]() {
				++colTablIdx;
				if (colTablIdx == colorTabl.size()) colTablIdx = 0;
			};
			auto& nodes = swc.getNodes();

			// build tree
			std::stack<std::pair<SWCIDTy, size_t>> stk;
			for (const auto& node : nodes)
				if (node.parentID != -1)
					tree[node.parentID].emplace_back(node.id);
				else
					stk.emplace(node.id, 0);

			// DFS to append to GLPathRenderer
			SWCNode curr;
			while (!stk.empty())
			{
				bool shouldDiveToLeaf = true;
				auto& [id, childIdx] = stk.top();
				curr.id = id;
				curr = *nodes.find(curr);
				if (curr.parentID == -1
					&& childIdx == 0)
				{
					// reach a root vert
					// and it has no children visited,
					// add a new path and new subPath
					pathRenderer.endPath();
					GLuint glID = pathRenderer.addPath(colorTabl[colTablIdx],
						glm::vec3{ curr.x,curr.y,curr.z });
					increaseColTablIdx();
					pathRenderer.startPath(glID);

					glID = pathRenderer.getPaths().at(glID).getRootID();
					idTabl[id] = glID;

					glID = pathRenderer.addSubPath();
					pathRenderer.startSubPath(glID);
				}
				else if (tree.find(id) != tree.end()
					&& childIdx < tree[id].size())
				{
					// reach a non-leaf vert (root vert)
					// and it still has children not visited
					// add a new subPath and start from this vert
					GLuint glID = idTabl[id];
					pathRenderer.startVertex(glID);
					glID = pathRenderer.addSubPath();
					pathRenderer.startSubPath(glID);
				}
				else
				{
					// reach a leaf node,
					// or all children of non-leaf node is visited, pop
					stk.pop();
					pathRenderer.endSubPath();
					shouldDiveToLeaf = false;
				}

				// add vert until reach a leaf
				if (shouldDiveToLeaf)
					do
					{
						auto& [id, childIdx] = stk.top();
						if (childIdx == tree[id].size())
						{
							++childIdx;
							break;
						}
						curr.id = tree[id][childIdx++];
						curr = *nodes.find(curr);

						GLuint vertID = pathRenderer.addVertex(
							glm::vec3{ curr.x,curr.y,curr.z });
						pathRenderer.startVertex(vertID);
						idTabl[curr.id] = vertID;

						stk.emplace(curr.id, 0);
					} while (tree.find(curr.id) != tree.end());
			}
		}
		static void fromGLPathRendererToSWC(const GLPathRenderer& pathRenderer, SWC& swc)
		{
			// id in GLPathRenderer -> ids of its children in GLPathRenderer
			std::unordered_map<GLuint, std::unordered_set<GLuint>> tree;
			// id in GLPathRenderer -> (swcID, swcID of parent)
			std::unordered_map<GLuint, std::pair<SWCIDTy, SWCIDTy>> swcTreeLinks;

			auto& vertPoss = pathRenderer.getVertexPositions();
			SWCIDTy guid = 0;
			swc.clear();
			for (const auto& [id, path] : pathRenderer.getPaths())
			{
				// build tree
				if (!tree.empty())
					throw std::runtime_error("GLPathRenderer didn't form trees!");
				for (const auto& [id, subPath] : path.getSubPaths())
				{
					GLuint curr = 1;
					const auto& vertIDs = subPath.getVertexIDs();
					for (; curr < vertIDs.size(); ++curr)
						tree[vertIDs[curr - 1]].emplace(vertIDs[curr]);
				}
				// BFS to gen SWCNode::id's
				swcTreeLinks.clear();
				{
					GLuint rootID = path.getRootID();
					swcTreeLinks[rootID].first = guid++;
					swcTreeLinks[rootID].second = -1; // root has no parent
				}
				while (!tree.empty())
				{
					auto [id, swcID, swcParID] = [&]() {
						// find vert has both SWCNode::id and links
						for (const auto& [id, swcPair] : swcTreeLinks)
							if (tree.find(id) != tree.end())
								return std::tuple{ id, swcPair.first, swcPair.second };
					}();
					auto& links = tree[id];
					for (GLuint linked : links)
					{
						swcTreeLinks[linked].first = guid++;
						swcTreeLinks[linked].second = swcID;
					}
					tree.erase(id);
				}
				// insert to SWC
				SWCNode swcNode;
				for (const auto& [id, swcPair] : swcTreeLinks )
				{
					swcNode.id = swcPair.first;
					swcNode.parentID = swcPair.second;
					swcNode.x = vertPoss[id].x;
					swcNode.y = vertPoss[id].y;
					swcNode.z = vertPoss[id].z;
					swcNode.radius = 1.0;
					swcNode.type = SWCNodeType::Axon;
					// the correct order of SWCNode::id will be maintained by SWC
					swc.add(swcNode);
				}
			}
		}
	};
}

#endif // !KOUEK_SWC_CONVERTER_H
