#include "lab_m2/tema2/tema2.h"


#include <iostream>

#include "pfd/portable-file-dialogs.h"

using namespace std;
using namespace m2;


/*
 *  To find out more about `FrameStart`, `Update`, `FrameEnd`
 *  and the order in which they are called, see `world.cpp`.
 */


Tema2::Tema2()
{
	outputMode = 0;
	gpuProcessing = false;
	saveScreenToImage = false;
	window->SetSize(600, 600);
}


Tema2::~Tema2()
{
}


void Tema2::Init()
{
	// Load default texture fore imagine processing
	originalImage = TextureManager::LoadTexture(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::TEXTURES, "cube", "pos_x.png"), nullptr, "image", true, true);
	processedImage = TextureManager::LoadTexture(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::TEXTURES, "cube", "pos_x.png"), nullptr, "newImage", true, true);

	{
		Mesh* mesh = new Mesh("quad");
		mesh->LoadMesh(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::MODELS, "primitives"), "quad.obj");
		mesh->UseMaterials(false);
		meshes[mesh->GetMeshID()] = mesh;
	}

	std::string shaderPath = PATH_JOIN(window->props.selfDir, SOURCE_PATH::M2, "tema2", "shaders");

	// Create a shader program for particle system
	{
		Shader* shader = new Shader("Tema2");
		shader->AddShader(PATH_JOIN(shaderPath, "VertexShader.glsl"), GL_VERTEX_SHADER);
		shader->AddShader(PATH_JOIN(shaderPath, "FragmentShader.glsl"), GL_FRAGMENT_SHADER);

		shader->CreateAndLink();
		shaders[shader->GetName()] = shader;
	}
}


void Tema2::FrameStart()
{
}


void Tema2::Update(float deltaTimeSeconds)
{
	ClearScreen();

	auto shader = shaders["Tema2"];
	shader->Use();

	if (saveScreenToImage)
	{
		window->SetSize(originalImage->GetWidth(), originalImage->GetHeight());
	}

	int flip_loc = shader->GetUniformLocation("flipVertical");
	glUniform1i(flip_loc, saveScreenToImage ? 0 : 1);

	int screenSize_loc = shader->GetUniformLocation("screenSize");
	glm::ivec2 resolution = window->GetResolution();
	glUniform2i(screenSize_loc, resolution.x, resolution.y);

	int outputMode_loc = shader->GetUniformLocation("outputMode");
	glUniform1i(outputMode_loc, outputMode);

	int locTexture = shader->GetUniformLocation("textureImage");
	glUniform1i(locTexture, 0);

	auto textureImage = (gpuProcessing == true) ? originalImage : processedImage;
	textureImage->BindToTextureUnit(GL_TEXTURE0);

	RenderMesh(meshes["quad"], shader, glm::mat4(1));

	if (saveScreenToImage)
	{
		saveScreenToImage = false;

		GLenum format = GL_RGB;
		if (originalImage->GetNrChannels() == 4)
		{
			format = GL_RGBA;
		}

		glReadPixels(0, 0, originalImage->GetWidth(), originalImage->GetHeight(), format, GL_UNSIGNED_BYTE, processedImage->GetImageData());
		processedImage->UploadNewData(processedImage->GetImageData());
		SaveImage("shader_processing_" + std::to_string(outputMode));

		float aspectRatio = static_cast<float>(originalImage->GetWidth()) / originalImage->GetHeight();
		window->SetSize(static_cast<int>(600 * aspectRatio), 600);
	}
}


void Tema2::FrameEnd()
{
	DrawCoordinateSystem();
}


void Tema2::OnFileSelected(const std::string& fileName)
{
	if (fileName.size())
	{
		std::cout << fileName << endl;
		originalImage = TextureManager::LoadTexture(fileName, nullptr, "image", true, true);
		processedImage = TextureManager::LoadTexture(fileName, nullptr, "newImage", true, true);

		float aspectRatio = static_cast<float>(originalImage->GetWidth()) / originalImage->GetHeight();
		window->SetSize(static_cast<int>(600 * aspectRatio), 600);
	}
}


void Tema2::Canny(unsigned char* originalData, unsigned char* proccessedData)
{
	unsigned int channels = originalImage->GetNrChannels();
	glm::ivec2 imageSize = glm::ivec2(originalImage->GetWidth(), originalImage->GetHeight());
	unsigned int dataSize = originalImage->GetWidth() * originalImage->GetHeight() * originalImage->GetNrChannels();

	unsigned char* blurData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* grayscaleData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* sobelData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* dirData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* nonmaxData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* thresholdData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* hysteresisData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);

	// grayscale
	memcpy(grayscaleData, originalData, dataSize);
	for (int i = 0; i < imageSize.y; i++)
	{
		for (int j = 0; j < imageSize.x; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			unsigned char value = static_cast<unsigned char>(originalData[offset + 0] * 0.2f + originalData[offset + 1] * 0.71f + originalData[offset + 2] * 0.07f);
			memset(&grayscaleData[offset], value, 3);
		}
	}

	// blur
	memcpy(blurData, grayscaleData, dataSize);
	for (int i = 1; i < imageSize.y - 1; i++)
	{
		for (int j = 1; j < imageSize.x - 1; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			glm::vec3 sum = glm::vec3(0);

			for (int r = i - 1; r <= i + 1; r++)
			{
				for (int c = j - 1; c <= j + 1; c++)
				{
					int off = channels * (r * imageSize.x + c);
					sum[0] += grayscaleData[off];
					sum[1] += grayscaleData[off + 1];
					sum[2] += grayscaleData[off + 2];
				}
			}

			float samples = pow((2 * 1 + 1), 2);
			memset(&blurData[offset], static_cast<unsigned char>(sum[0] / samples), 1);
			memset(&blurData[offset + 1], static_cast<unsigned char>(sum[1] / samples), 1);
			memset(&blurData[offset + 2], static_cast<unsigned char>(sum[2] / samples), 1);
		}
	}

	// sobel
	int dx[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	int dy[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	memcpy(sobelData, blurData, dataSize);
	memset(dirData, 0, dataSize);
	for (int i = 1; i < imageSize.y - 1; i++)
	{
		for (int j = 1; j < imageSize.x - 1; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			float sumx = 0.0, sumy = 0.0;
			int s = 0;

			for (int r = i - 1; r <= i + 1; r++)
			{
				for (int c = j - 1; c <= j + 1; c++)
				{
					int off = channels * (r * imageSize.x + c);
					sumx += dx[s] * blurData[off];
					sumy += dy[s] * blurData[off];
					s++;
				}
			}

			// compute amplitude
			float amp = sqrt(sumx * sumx + sumy * sumy);
			memset(&sobelData[offset], static_cast<unsigned char>(amp), 3);

			// compute gradient direction and adjust it
			float dir;
			if (abs(sumy) <= 0.001)
				dir = 0;
			else if (abs(sumx) <= 0.001)
				dir = 90;
			else
			{
				dir = atan2(sumy, sumx) * 180 / M_PI;
				if (dir < 0)
					dir += 180;

				if (0 <= dir && dir < 22.5)
					dir = 0;
				else if (22.5 <= dir && dir < 67.5)
					dir = 45;
				else if (67.5 <= dir && dir < 112.5)
					dir = 90;
				else if (112.5 <= dir && dir < 157.5)
					dir = 135;
				else if (157.5 <= dir && dir <= 180)
					dir = 0;
			}
			memset(&dirData[offset], static_cast<unsigned char>(dir), 3);
		}
	}

	// non-maxima suppresion
	memcpy(nonmaxData, sobelData, dataSize);
	for (int i = 1; i < imageSize.y - 1; i++)
	{
		for (int j = 1; j < imageSize.x - 1; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			float dir = dirData[offset];
			float amp1 = 255, amp2 = 255;

			if (dir == 0)
			{
				amp1 = sobelData[channels * (i * imageSize.x + (j + 1))];
				amp2 = sobelData[channels * (i * imageSize.x + (j - 1))];
			}
			else if (dir == 45)
			{
				amp1 = sobelData[channels * ((i + 1) * imageSize.x + (j - 1))];
				amp2 = sobelData[channels * ((i - 1) * imageSize.x + (j + 1))];
			}
			else if (dir == 90)
			{
				amp1 = sobelData[channels * ((i + 1) * imageSize.x + j)];
				amp2 = sobelData[channels * ((i - 1) * imageSize.x + j)];
			}
			else if (dir == 135)
			{
				amp1 = sobelData[channels * ((i - 1) * imageSize.x + (j - 1))];
				amp2 = sobelData[channels * ((i + 1) * imageSize.x + (j + 1))];
			}

			float nonmax;
			if (sobelData[offset] >= amp1 && sobelData[offset] >= amp2)
				nonmax = sobelData[offset];
			else
				nonmax = 0;
			memset(&nonmaxData[offset], static_cast<unsigned char>(nonmax), 3);
		}
	}

	// threshold
	memcpy(thresholdData, nonmaxData, dataSize);
	for (int i = 0; i < imageSize.y; i++)
	{
		for (int j = 0; j < imageSize.x; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			if (nonmaxData[offset] < 75)
				memset(&thresholdData[offset], 0, 3);
			else if (nonmaxData[offset] > 150)
				memset(&thresholdData[offset], 255, 3);
			else
				memset(&thresholdData[offset], nonmaxData[offset], 3);
		}
	}

	// hysteresis
	memcpy(hysteresisData, thresholdData, dataSize);
	for (int i = 1; i < imageSize.y - 1; i++)
	{
		for (int j = 1; j < imageSize.x - 1; j++)
		{
			int offset = channels * (i * imageSize.x + j);

			if (75 <= hysteresisData[offset] && hysteresisData[offset] <= 150)
			{
				if (hysteresisData[channels * ((i + 1) * imageSize.x + (j - 1))] == 255 ||
					hysteresisData[channels * ((i + 1) * imageSize.x + j)] == 255 ||
					hysteresisData[channels * ((i + 1) * imageSize.x + (j + 1))] == 255 ||
					hysteresisData[channels * (i * imageSize.x + (j - 1))] == 255 ||
					hysteresisData[channels * (i * imageSize.x + (j + 1))] == 255 ||
					hysteresisData[channels * ((i - 1) * imageSize.x + (j - 1))] == 255 ||
					hysteresisData[channels * ((i - 1) * imageSize.x + j)] == 255 ||
					hysteresisData[channels * ((i - 1) * imageSize.x + (j + 1))] == 255)
					memset(&hysteresisData[offset], 255, 3);
				else
					memset(&hysteresisData[offset], 0, 3);
			}
		}
	}

	memcpy(proccessedData, hysteresisData, dataSize);
}


void Tema2::MedianCut(unsigned char* originalData, unsigned char* proccessedData, int n)
{
	unsigned int channels = originalImage->GetNrChannels();
	glm::ivec2 imageSize = glm::ivec2(originalImage->GetWidth(), originalImage->GetHeight());
	unsigned int dataSize = originalImage->GetWidth() * originalImage->GetHeight() * originalImage->GetNrChannels();

	unsigned char* recolorData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);

	// median cut
	int redMin = 255, redMax = 0;
	int greenMin = 255, greenMax = 0;
	int blueMin = 255, blueMax = 0;

	unsigned char* red = (unsigned char*)malloc(sizeof(unsigned char) * imageSize.x * imageSize.y);
	unsigned char* green = (unsigned char*)malloc(sizeof(unsigned char) * imageSize.x * imageSize.y);
	unsigned char* blue = (unsigned char*)malloc(sizeof(unsigned char) * imageSize.x * imageSize.y);
	unsigned int* position = (unsigned int*)malloc(sizeof(unsigned int) * imageSize.x * imageSize.y);
	int index = 0;

	// find the color channel with max difference
	for (int i = 0; i < imageSize.y; i++)
	{
		for (int j = 0; j < imageSize.x; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			position[index] = offset;

			if (originalData[offset] < redMin)
				redMin = originalData[offset];
			if (originalData[offset] > redMax)
				redMax = originalData[offset];
			red[index] = originalData[offset];

			if (originalData[offset + 1] < greenMin)
				greenMin = originalData[offset + 1];
			if (originalData[offset + 1] > greenMax)
				greenMax = originalData[offset + 1];
			green[index] = originalData[offset + 1];

			if (originalData[offset + 2] < blueMin)
				blueMin = originalData[offset + 2];
			if (originalData[offset + 2] > blueMax)
				blueMax = originalData[offset + 2];
			blue[index] = originalData[offset + 2];

			index++;
		}
	}

	int sortBy;
	int redDiff = redMax - redMin;
	int greenDiff = greenMax - greenMin;
	int blueDiff = blueMax - blueMin;
	if (redDiff >= greenDiff && redDiff >= blueDiff)
		sortBy = 0;
	else if (greenDiff > redDiff && greenDiff > blueDiff)
		sortBy = 1;
	else if (blueDiff > redDiff && blueDiff > greenDiff)
		sortBy = 2;

	// sort pixels
	for (int i = 0; i < index; i++)
	{
		for (int j = i + 1; j < index; j++)
		{
			if ((sortBy == 0 && red[j] < red[i]) ||
				(sortBy == 1 && green[j] < green[i]) ||
				(sortBy == 2 && blue[j] < blue[i]))
			{
				unsigned char aux;
				unsigned int pos;

				aux = red[i];
				red[i] = red[j];
				red[j] = aux;

				aux = green[i];
				green[i] = green[j];
				green[j] = aux;

				aux = blue[i];
				blue[i] = blue[j];
				blue[j] = aux;

				pos = position[i];
				position[i] = position[j];
				position[j] = pos;
			}
		}
	}

	int interval = imageSize.x * imageSize.y / n;

	// compute medium values for each interval
	for (int i = 0; i < n; i++)
	{
		float sumRed = 0.0;
		float sumGreen = 0.0;
		float sumBlue = 0.0;

		for (int j = i * interval; j < i * interval + interval; j++)
		{
			sumRed += red[j];
			sumGreen += green[j];
			sumBlue += blue[j];
		}

		sumRed = sumRed / interval;
		sumGreen = sumGreen / interval;
		sumBlue = sumBlue / interval;

		for (int j = i * interval; j < i * interval + interval; j++)
		{
			red[j] = sumRed;
			green[j] = sumGreen;
			blue[j] = sumBlue;
		}
	}

	// change the original pixels
	for (int i = 0; i < index; i++)
	{
		recolorData[position[i]] = red[i];
		recolorData[position[i] + 1] = green[i];
		recolorData[position[i] + 2] = blue[i];
	}

	memcpy(proccessedData, recolorData, dataSize);
}


void Tema2::MedianFilter(unsigned char* originalData, unsigned char* proccessedData)
{
	unsigned int channels = originalImage->GetNrChannels();
	glm::ivec2 imageSize = glm::ivec2(originalImage->GetWidth(), originalImage->GetHeight());
	unsigned int dataSize = originalImage->GetWidth() * originalImage->GetHeight() * originalImage->GetNrChannels();

	unsigned char* grayscaleData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* medianData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);

	// grayscale
	memcpy(grayscaleData, originalData, dataSize);
	for (int i = 0; i < imageSize.y; i++)
	{
		for (int j = 0; j < imageSize.x; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			unsigned char value = static_cast<unsigned char>(originalData[offset + 0] * 0.2f + originalData[offset + 1] * 0.71f + originalData[offset + 2] * 0.07f);
			memset(&grayscaleData[offset], value, 3);
		}
	}

	// median filter
	memcpy(medianData, originalData, dataSize);
	for (int i = 1; i < imageSize.y - 1; i++)
	{
		for (int j = 1; j < imageSize.x - 1; j++)
		{
			int offset = channels * (i * imageSize.x + j);
			glm::vec3 neighColors[9];
			unsigned char neighValues[9];
			int ind = 0;

			for (int r = i - 1; r <= i + 1; r++)
			{
				for (int c = j - 1; c <= j + 1; c++)
				{
					int off = channels * (r * imageSize.x + c);
					neighColors[ind] = glm::vec3(originalData[off], originalData[off + 1], originalData[off + 2]);
					neighValues[ind] = grayscaleData[off];
					ind++;
				}
			}

			for (int k = 0; k < 9; k++)
			{
				for (int l = k; l < 9; l++)
				{
					if (neighValues[k] > neighValues[l])
					{
						glm::vec3 vaux;
						unsigned char aux;

						vaux = neighColors[k];
						neighColors[k] = neighColors[l];
						neighColors[l] = vaux;

						aux = neighValues[k];
						neighValues[k] = neighValues[l];
						neighValues[l] = aux;
					}
				}
			}

			medianData[offset] = neighColors[4].x;
			medianData[offset + 1] = neighColors[4].y;
			medianData[offset + 2] = neighColors[4].z;
		}
	}

	memcpy(proccessedData, medianData, dataSize);
}


void Tema2::PhotoToCartoon(int type)
{
	unsigned int channels = originalImage->GetNrChannels();
	if (channels < 3)
		return;

	glm::ivec2 imageSize = glm::ivec2(originalImage->GetWidth(), originalImage->GetHeight());
	unsigned int dataSize = originalImage->GetWidth() * originalImage->GetHeight() * originalImage->GetNrChannels();

	unsigned char* originalData = originalImage->GetImageData();

	unsigned char* edgeData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* recolorData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* medianData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);
	unsigned char* finalData = (unsigned char*)malloc(sizeof(unsigned char) * dataSize);

	int noColors = 16;

	if (type == 1)
	{
		Canny(originalData, edgeData);
		processedImage->UploadNewData(edgeData);
		return;
	}

	if (type == 2)
	{
		MedianCut(originalData, recolorData, noColors);
		processedImage->UploadNewData(recolorData);
		return;
	}

	if (type == 3)
	{
		MedianFilter(originalData, medianData);
		processedImage->UploadNewData(medianData);
		return;
	}

	if (type == 4)
	{
		Canny(originalData, edgeData);
		MedianCut(originalData, recolorData, noColors);
		MedianFilter(recolorData, medianData);

		// combine the proccessed images
		memcpy(finalData, medianData, dataSize);
		for (int i = 0; i < imageSize.y; i++)
		{
			for (int j = 0; j < imageSize.x; j++)
			{
				int offset = channels * (i * imageSize.x + j);

				if (edgeData[offset] == 255)
				{
					finalData[offset] = 0;
					finalData[offset + 1] = 0;
					finalData[offset + 2] = 0;
				}
			}
		}

		processedImage->UploadNewData(finalData);
	}
}


void Tema2::SaveImage(const std::string& fileName)
{
	cout << "Saving image! ";
	processedImage->SaveToFile((fileName + ".png").c_str());
	cout << "[Done]" << endl;
}


void Tema2::OpenDialog()
{
	std::vector<std::string> filters =
	{
		"Image Files", "*.png *.jpg *.jpeg *.bmp",
		"All Files", "*"
	};

	auto selection = pfd::open_file("Select a file", ".", filters).result();
	if (!selection.empty())
	{
		std::cout << "User selected file " << selection[0] << "\n";
		OnFileSelected(selection[0]);
	}
}


/*
 *  These are callback functions. To find more about callbacks and
 *  how they behave, see `input_controller.h`.
 */


void Tema2::OnInputUpdate(float deltaTime, int mods)
{
	// Treat continuous update based on input
}


void Tema2::OnKeyPress(int key, int mods)
{
	// Add key press event
	if (key == GLFW_KEY_F || key == GLFW_KEY_ENTER || key == GLFW_KEY_SPACE)
	{
		OpenDialog();
	}

	if (key - GLFW_KEY_0 >= 0 && key <= GLFW_KEY_3)
	{
		outputMode = key - GLFW_KEY_0;

		if (gpuProcessing == false)
		{
			outputMode = 0;
			if (key == GLFW_KEY_0)
				processedImage->UploadNewData(originalImage->GetImageData());
		}
	}

	// frontier detection
	if (key == GLFW_KEY_1)
		PhotoToCartoon(1);
	// color reduce
	if (key == GLFW_KEY_2)
		PhotoToCartoon(2);
	// median
	if (key == GLFW_KEY_3)
		PhotoToCartoon(3);
	// final image
	if (key == GLFW_KEY_4)
		PhotoToCartoon(4);

	if (key == GLFW_KEY_S && mods & GLFW_MOD_CONTROL)
	{
		if (!gpuProcessing)
		{
			SaveImage("processCPU_" + std::to_string(outputMode));
		}
		else {
			saveScreenToImage = true;
		}
	}
}


void Tema2::OnKeyRelease(int key, int mods)
{
	// Add key release event
}


void Tema2::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
	// Add mouse move event
}


void Tema2::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
	// Add mouse button press event
}


void Tema2::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
	// Add mouse button release event
}


void Tema2::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
	// Treat mouse scroll event
}


void Tema2::OnWindowResize(int width, int height)
{
	// Treat window resize event
}
